"""
Product data extractor: programmatic hydration + LLM gap-fill.

Three stages:
  A) Walk parsed structured data to fill Product fields (free)
  B) Call LLM only for missing fields — primarily category (cheap)
  C) Validate with Pydantic, retry category with taxonomy subset on failure
"""

import difflib
import html as html_lib
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ValidationError

import ai
from models import VALID_CATEGORIES, Category, Price, Product, Variant
from parser import ParsedPage

logger = logging.getLogger(__name__)

# Keys that signal "this dict is about a product"
PRODUCT_SIGNAL_KEYS = {
    "name", "title", "productName", "fullTitle",
    "price", "prices", "priceAsNumber", "offers",
    "description", "descriptionHtml", "productDescription",
    "brand", "brandName",
    "sku", "productSku",
    "image", "images", "media", "mediaObjects", "contentImages", "productImages",
    "variants", "items", "skus", "sizes", "hasVariant",
    "google_merchant_category", "productInfo", "productDetails",
    "color", "colorDescription", "colors", "questions",
}

# Deeper keyword-to-prefix mapping for taxonomy narrowing.
# Ordered most-specific to least-specific; first match wins.
# When a keyword matches, we send only categories under that prefix to the LLM.
_TAXONOMY_PREFIX_MAP: list[tuple[str, list[str]]] = [
    # Hardware - specific tool types
    ("Hardware > Tools > Drills", ["drill", "driver kit", "impact driver"]),
    ("Hardware > Tools > Saws", ["chainsaw", "jigsaw", "circular saw", "miter saw", "table saw", "band saw"]),
    ("Hardware > Tools > Sanders", ["sander"]),
    ("Hardware > Tools > Grinders", ["grinder", "angle grinder"]),
    ("Hardware > Tools > Nailers & Staplers", ["nailer", "nail gun", "staple gun"]),
    ("Hardware > Tools", ["power tool", "wrench", "screwdriver", "hammer", "plier", "tool set", "tool kit"]),
    ("Hardware", ["hardware", "plumbing"]),
    # Home & Garden
    ("Home & Garden > Lighting", ["lamp", "lighting", "chandelier", "sconce", "lantern", "pendant light"]),
    ("Home & Garden > Kitchen & Dining", ["kitchen", "cookware", "bakeware", "dinnerware"]),
    ("Home & Garden > Lawn & Garden", ["lawn mower", "garden tool", "trimmer"]),
    ("Home & Garden", ["furniture", "garden", "rug", "pillow", "curtain", "bedding", "decor"]),
    # Apparel & Accessories
    ("Apparel & Accessories > Shoes", ["shoe", "sneaker", "boot", "sandal", "slipper", "loafer"]),
    ("Apparel & Accessories > Clothing", ["shirt", "henley", "pant", "trouser", "dress", "jacket", "sweater", "coat", "hoodie", "polo"]),
    ("Apparel & Accessories", ["clothing", "apparel", "wear", "accessory"]),
    # Electronics
    ("Electronics > Audio", ["headphone", "speaker", "earbuds"]),
    ("Electronics > Computers", ["laptop", "computer", "tablet"]),
    ("Electronics", ["phone", "camera", "electronic"]),
    # Sporting Goods
    ("Sporting Goods > Exercise & Fitness", ["treadmill", "elliptical", "exercise bike"]),
    ("Sporting Goods", ["sport", "fitness", "gym", "ball", "racket", "bicycle", "bike"]),
]

# Pre-built lookup: lowercase -> exact category (for case-insensitive matching)
_CATEGORY_LOWER = {c.lower(): c for c in VALID_CATEGORIES}

# Pre-built lookup: leaf term -> list of categories containing it
_CATEGORY_BY_LEAF: dict[str, list[str]] = {}
for _cat in VALID_CATEGORIES:
    _leaf = _cat.rsplit(" > ", 1)[-1].lower()
    _CATEGORY_BY_LEAF.setdefault(_leaf, []).append(_cat)

# Model to use for LLM gap-filling
LLM_MODEL = "google/gemini-2.0-flash-lite-001"


def _fuzzy_match_category(raw: str) -> str | None:
    """Try to match a raw category string to a valid taxonomy entry without LLM.

    Strategies tried in order:
    1. Exact match
    2. Case-insensitive match
    3. Leaf-term match (most specific segment matches a taxonomy leaf)
    4. Path-aware difflib (match within the guessed top-level subset)
    Returns None if no confident match found.
    """
    if not raw:
        return None

    # 1. Exact match
    if raw in VALID_CATEGORIES:
        return raw

    # 2. Case-insensitive
    lower = raw.lower()
    if lower in _CATEGORY_LOWER:
        return _CATEGORY_LOWER[lower]

    # 3. Leaf-term match: take the LLM's most specific term and find categories ending with it
    leaf = raw.rsplit(" > ", 1)[-1].strip().lower()
    if leaf in _CATEGORY_BY_LEAF:
        candidates = _CATEGORY_BY_LEAF[leaf]
        if len(candidates) == 1:
            return candidates[0]
        # Multiple candidates — pick the one whose path best matches the LLM's output
        best = difflib.get_close_matches(raw, candidates, n=1, cutoff=0.4)
        if best:
            return best[0]
        # Just return the most specific (longest path)
        return max(candidates, key=len)

    # 4. Path-aware difflib (match within top-level subset for better results)
    top_parts = raw.split(" > ")
    if top_parts:
        top_guess = top_parts[0]
        subset = [c for c in VALID_CATEGORIES if c.startswith(top_guess)]
        if subset:
            matches = difflib.get_close_matches(raw, subset, n=1, cutoff=0.5)
            if matches:
                return matches[0]

    # Full taxonomy difflib as absolute last resort
    matches = difflib.get_close_matches(raw, VALID_CATEGORIES, n=1, cutoff=0.6)
    if matches:
        return matches[0]

    return None


# ===== Metrics =====

PRODUCT_FIELDS = ["name", "brand", "price", "description", "key_features",
                  "image_urls", "video_url", "category", "colors", "variants"]


@dataclass
class ExtractionMetrics:
    """Per-file metrics collected during extraction."""
    filename: str = ""
    # Stage timing (seconds)
    hydrate_time: float = 0.0
    llm_fill_time: float = 0.0
    validate_time: float = 0.0
    total_time: float = 0.0
    # Field provenance: which fields came from parsing vs LLM
    fields_from_parser: list[str] = field(default_factory=list)
    fields_from_llm: list[str] = field(default_factory=list)
    fields_missing_after_all: list[str] = field(default_factory=list)
    # LLM usage
    llm_calls: int = 0  # 0 = all from parser, 1 = gap-fill, 2 = gap-fill + category retry
    llm_skipped: bool = False  # True if all fields hydrated, no LLM needed
    # Validation
    category_first_attempt_valid: bool = False
    category_fuzzy_matched: bool = False  # resolved by fuzzy match (no LLM retry)
    category_retry_needed: bool = False
    category_retry_succeeded: bool = False
    # How category was resolved: "parser", "llm_exact", "fuzzy_match", "llm_retry", "failed"
    category_resolution: str = ""
    validation_error: str | None = None
    # Output richness
    num_variants: int = 0
    num_images: int = 0
    num_features: int = 0
    num_colors: int = 0
    has_video: bool = False
    has_sale_price: bool = False


# ===== LLM Response Schema =====

class LLMGapFill(BaseModel):
    category: str | None = None
    key_features: list[str] | None = None
    colors: list[str] | None = None
    description: str | None = None
    price: float | None = None
    currency: str | None = None


# ===== Main Entry Point =====

async def extract_product(parsed: ParsedPage, filename: str) -> tuple[Product, ExtractionMetrics]:
    """Full extraction pipeline: hydrate -> LLM fill -> validate. Returns product and metrics."""
    metrics = ExtractionMetrics(filename=filename)
    t_start = time.monotonic()

    # Stage A: Programmatic hydration (free)
    t0 = time.monotonic()
    fields = _hydrate_fields(parsed)
    metrics.hydrate_time = time.monotonic() - t0

    # Record what parsing alone filled
    for f in PRODUCT_FIELDS:
        val = fields.get(f)
        if val is not None and val != "" and val != []:
            metrics.fields_from_parser.append(f)

    # Stage B: LLM gap-fill (cheap, only for missing fields)
    t0 = time.monotonic()
    fields, llm_calls, llm_skipped = await _llm_fill_gaps(fields, parsed)
    metrics.llm_fill_time = time.monotonic() - t0
    metrics.llm_calls = llm_calls
    metrics.llm_skipped = llm_skipped

    # Record what LLM added
    for f in PRODUCT_FIELDS:
        val = fields.get(f)
        if val is not None and val != "" and val != []:
            if f not in metrics.fields_from_parser:
                metrics.fields_from_llm.append(f)

    # Stage C: Validate with Pydantic, retry category if needed
    t0 = time.monotonic()
    product, cat_first_valid, cat_fuzzy, cat_retry_needed, cat_retry_ok = await _validate_product(fields, parsed)
    metrics.validate_time = time.monotonic() - t0
    metrics.category_first_attempt_valid = cat_first_valid
    metrics.category_fuzzy_matched = cat_fuzzy
    metrics.category_retry_needed = cat_retry_needed
    metrics.category_retry_succeeded = cat_retry_ok
    if cat_retry_needed:
        metrics.llm_calls += 1

    # Determine how category was resolved
    if "category" in metrics.fields_from_parser:
        metrics.category_resolution = "parser"
    elif cat_first_valid and not cat_fuzzy:
        metrics.category_resolution = "llm_exact"
    elif cat_fuzzy:
        metrics.category_resolution = "fuzzy_match"
    elif cat_retry_ok:
        metrics.category_resolution = "llm_retry"
    else:
        metrics.category_resolution = "failed"

    # Output richness
    metrics.num_variants = len(product.variants)
    metrics.num_images = len(product.image_urls)
    metrics.num_features = len(product.key_features)
    metrics.num_colors = len(product.colors)
    metrics.has_video = product.video_url is not None
    metrics.has_sale_price = product.price.compare_at_price is not None

    # Fields still empty in final output
    for f in PRODUCT_FIELDS:
        val = getattr(product, f, None)
        if val is None or val == "" or val == []:
            metrics.fields_missing_after_all.append(f)

    metrics.total_time = time.monotonic() - t_start
    return product, metrics


# =====================================================================
# Stage A: Programmatic Hydration
# =====================================================================

def _hydrate_fields(parsed: ParsedPage) -> dict:
    """Walk all data sources in priority order to fill Product fields."""
    fields: dict[str, Any] = {}

    # 1. Embedded JSON (richest source for most pages)
    for source_name, data in parsed.embedded_json.items():
        product_obj = _find_product_object(data)
        if product_obj:
            _extract_fields_from_object(product_obj, fields)

    # 2. JSON-LD (standardized, reliable)
    for ld_block in parsed.json_ld:
        _extract_fields_from_json_ld(ld_block, fields)

    # 3. OG meta tags (fallback for name, description, image)
    _extract_fields_from_og(parsed.og_tags, fields)

    # 4. Body text price fallback (when price isn't in structured data)
    if "price" not in fields:
        _extract_price_from_text(parsed.body_text, fields)

    # 5. Look for colors in wider data structures (colorwayImages etc.)
    _extract_colors_from_all_data(parsed.embedded_json, fields)

    # 6. Collected image/video URLs from parser as final fallback
    if "image_urls" not in fields or not fields["image_urls"]:
        fields["image_urls"] = parsed.image_urls
    elif parsed.image_urls:
        # Merge parser-level images with already-found ones
        existing = set(fields["image_urls"])
        for url in parsed.image_urls:
            if url not in existing:
                fields["image_urls"].append(url)

    if "video_url" not in fields and parsed.video_urls:
        fields["video_url"] = parsed.video_urls[0]

    # 7. Breadcrumbs as category hint
    if "category" not in fields and parsed.breadcrumbs:
        fields.setdefault("_category_hints", []).append(" > ".join(parsed.breadcrumbs))

    return fields


# ----- Product Object Finder -----

def _find_product_object(data: Any, depth: int = 0) -> dict | None:
    """Recursively find the most product-like dict in an arbitrary JSON structure."""
    if depth > 10:
        return None

    if isinstance(data, dict):
        score = len(PRODUCT_SIGNAL_KEYS & set(data.keys()))
        best = data if score >= 3 else None
        best_score = score if score >= 3 else 0

        for v in data.values():
            child = _find_product_object(v, depth + 1)
            if child is not None:
                child_score = len(PRODUCT_SIGNAL_KEYS & set(child.keys()))
                if child_score > best_score:
                    best = child
                    best_score = child_score
        return best

    elif isinstance(data, list):
        best = None
        best_score = 0
        for item in data:
            child = _find_product_object(item, depth + 1)
            if child is not None:
                child_score = len(PRODUCT_SIGNAL_KEYS & set(child.keys()))
                if child_score > best_score:
                    best = child
                    best_score = child_score
        return best

    return None


# ----- Field Extractors from Embedded JSON -----

def _extract_fields_from_object(obj: dict, fields: dict) -> None:
    """Extract Product fields from a product-like embedded JSON object."""
    _extract_name(obj, fields)
    _extract_brand(obj, fields)
    _extract_price_field(obj, fields)
    _extract_description(obj, fields)
    _extract_features(obj, fields)
    _extract_images_from_obj(obj, fields)
    _extract_video_from_obj(obj, fields)
    _extract_colors(obj, fields)
    _extract_variants(obj, fields)
    _extract_category_hint(obj, fields)


def _extract_name(obj: dict, fields: dict) -> None:
    if "name" in fields:
        return
    name = (
        obj.get("name")
        or obj.get("title")
        or obj.get("productName")
        or obj.get("fullTitle")
    )
    # Check nested productInfo
    if not name:
        pi = obj.get("productInfo")
        if isinstance(pi, dict):
            name = pi.get("fullTitle") or pi.get("title")
    # Check nested content
    if not name:
        content = obj.get("content")
        if isinstance(content, dict):
            name = content.get("productName")
    if name and isinstance(name, str):
        fields["name"] = html_lib.unescape(name.strip())


def _extract_brand(obj: dict, fields: dict) -> None:
    if "brand" in fields:
        return
    brand = obj.get("brandName") or obj.get("brand")
    if isinstance(brand, dict):
        brand = brand.get("name")
    if isinstance(brand, list):
        brand = brand[0] if brand else None
    if brand and isinstance(brand, str):
        fields["brand"] = brand.strip()


def _extract_price_field(obj: dict, fields: dict) -> None:
    if "price" in fields:
        return

    # Pattern: priceAsNumber + currency in price string
    if "priceAsNumber" in obj:
        amount = float(obj["priceAsNumber"])
        currency = "USD"
        price_str = obj.get("price", "")
        if isinstance(price_str, str) and " " in price_str:
            currency = price_str.split()[-1]
        compare = None
        before = obj.get("priceBeforeDiscountAsNumber")
        if before and float(before) != amount:
            compare = float(before)
        fields["price"] = Price(price=amount, currency=currency, compare_at_price=compare)
        return

    # Pattern: prices dict with currentPrice/initialPrice
    prices = obj.get("prices")
    if isinstance(prices, dict) and "currentPrice" in prices:
        current = float(prices["currentPrice"])
        initial = prices.get("initialPrice")
        currency = prices.get("currency", "USD")
        compare = float(initial) if initial and float(initial) != current else None
        fields["price"] = Price(price=current, currency=currency, compare_at_price=compare)
        return

    # Pattern: prices list with isFullPrice flag
    if isinstance(prices, list) and prices:
        full_entry = next((p for p in prices if p.get("isFullPrice")), None)
        sale_entry = next((p for p in prices if not p.get("isFullPrice")), None)
        if full_entry:
            if sale_entry:
                fields["price"] = Price(
                    price=float(sale_entry["amount"]),
                    currency="USD",
                    compare_at_price=float(full_entry["amount"]),
                )
            else:
                fields["price"] = Price(price=float(full_entry["amount"]), currency="USD")
            return

    # Pattern: price dict with msrp
    price_obj = obj.get("price")
    if isinstance(price_obj, dict) and ("price" in price_obj or "catalogListPrice" in price_obj):
        amount = price_obj.get("price") or price_obj.get("catalogListPrice")
        msrp = price_obj.get("msrp")
        # Check for instant savings price
        after_savings = obj.get("priceAfterInstantSavings")
        if after_savings:
            actual_price = float(after_savings)
            compare = float(msrp) if msrp and float(msrp) != actual_price else None
        else:
            actual_price = float(amount) if amount else None
            compare = float(msrp) if msrp and amount and float(msrp) != float(amount) else None
        if actual_price:
            fields["price"] = Price(
                price=actual_price,
                currency=price_obj.get("priceCurrency", "USD"),
                compare_at_price=compare,
            )
            return

    # Pattern: JSON-LD offers
    offers = obj.get("offers")
    if isinstance(offers, dict) and "price" in offers:
        fields["price"] = Price(
            price=float(offers["price"]),
            currency=offers.get("priceCurrency", "USD"),
        )


def _extract_description(obj: dict, fields: dict) -> None:
    if "description" in fields:
        return
    desc = obj.get("description") or obj.get("descriptionHtml") or obj.get("excerpt")
    # Check nested productInfo
    if not desc:
        pi = obj.get("productInfo")
        if isinstance(pi, dict):
            desc = pi.get("productDescription")
    # Check nested content
    if not desc:
        content = obj.get("content")
        if isinstance(content, dict):
            desc = content.get("productFullDescription") or content.get("productShortDescription")
    # Check productDetails.details for description sections
    # Sections may have None headers, so look for sections with prose-length content
    if not desc:
        pd = obj.get("productDetails")
        if isinstance(pd, dict):
            details_list = pd.get("details", [])
            if isinstance(details_list, list):
                for section in details_list:
                    if isinstance(section, dict):
                        detail_items = section.get("details", [])
                        if isinstance(detail_items, list) and detail_items:
                            # Look for sections with longer text (descriptions, not bullet features)
                            combined = " ".join(str(d) for d in detail_items)
                            if len(combined) > 50 and not desc:
                                desc = combined
    if desc and isinstance(desc, str):
        # Clean up HTML in description if present
        desc = re.sub(r"<[^>]+>", " ", desc)
        desc = re.sub(r"\s+", " ", desc).strip()
        fields["description"] = desc


def _extract_features(obj: dict, fields: dict) -> None:
    if "key_features" in fields and fields["key_features"]:
        return
    features: list[str] = []

    # Pattern: positiveNotes array
    notes = obj.get("positiveNotes")
    if isinstance(notes, list):
        features.extend(str(n) for n in notes if n)

    # Pattern: featuresAndBenefits / productDetails in productInfo
    pi = obj.get("productInfo")
    if isinstance(pi, dict):
        for key in ("featuresAndBenefits", "productDetails"):
            sections = pi.get(key, [])
            if isinstance(sections, list):
                for section in sections:
                    if isinstance(section, dict):
                        body = section.get("body", [])
                        if isinstance(body, list):
                            features.extend(str(b) for b in body if b)

    # Pattern: productDetails.details sections
    # Sections have name/title/details — headers may be None, so extract from all
    pd = obj.get("productDetails")
    if isinstance(pd, dict):
        details_list = pd.get("details", [])
        if isinstance(details_list, list):
            for section in details_list:
                if isinstance(section, dict):
                    detail_items = section.get("details", [])
                    if isinstance(detail_items, list):
                        features.extend(str(d) for d in detail_items if d)

    # Pattern: properties with feature in attributeFQN
    props = obj.get("properties")
    if isinstance(props, list):
        for p in props:
            if isinstance(p, dict):
                fqn = str(p.get("attributeFQN", ""))
                if "feature" in fqn.lower() and "text" in fqn.lower():
                    vals = p.get("values", [])
                    if isinstance(vals, list) and vals:
                        val = vals[0].get("value") if isinstance(vals[0], dict) else vals[0]
                        if val:
                            features.append(str(val))

    # Pattern: bullet-formatted description
    desc = obj.get("description", "")
    if isinstance(desc, str) and "\n-" in desc and not features:
        for line in desc.split("\n"):
            line = line.strip().lstrip("- ").strip()
            if line and len(line) > 3:
                features.append(line)

    if features:
        fields["key_features"] = features


def _extract_images_from_obj(obj: dict, fields: dict) -> None:
    if "image_urls" in fields and fields["image_urls"]:
        return
    urls: list[str] = []

    # Pattern: mediaObjects with sources dict
    # Structure: mediaObjects[].sources.{mini,thumb,standard,full,max}[].url
    media_objs = obj.get("mediaObjects")
    if isinstance(media_objs, list):
        for mo in media_objs:
            if not isinstance(mo, dict):
                continue
            sources = mo.get("sources")
            if isinstance(sources, dict):
                # Prefer highest resolution
                for size_key in ("max", "full", "standard"):
                    variants = sources.get(size_key)
                    if isinstance(variants, list) and variants:
                        first = variants[0]
                        if isinstance(first, dict) and first.get("url"):
                            urls.append(first["url"])
                            break

    # Pattern: contentImages with properties
    content_imgs = obj.get("contentImages")
    if isinstance(content_imgs, list):
        for ci in content_imgs:
            if isinstance(ci, dict):
                props = ci.get("properties", {})
                if isinstance(props, dict):
                    for key in ("squarish", "portrait"):
                        img = props.get(key, {})
                        if isinstance(img, dict) and img.get("url"):
                            urls.append(img["url"])
                            break

    # Pattern: media list with src
    media_list = obj.get("media")
    if isinstance(media_list, list):
        for m in media_list:
            if isinstance(m, dict) and m.get("src"):
                urls.append(m["src"])

    # Pattern: content.productImages
    content = obj.get("content")
    if isinstance(content, dict):
        prod_imgs = content.get("productImages")
        if isinstance(prod_imgs, list):
            for pi in prod_imgs:
                if isinstance(pi, dict):
                    url = pi.get("imageUrl") or pi.get("src")
                    if url:
                        urls.append(url)

    # Pattern: images array
    images = obj.get("images")
    if isinstance(images, list):
        for img in images:
            if isinstance(img, str):
                urls.append(img)

    # Pattern: single image string (JSON-LD)
    image = obj.get("image")
    if isinstance(image, str):
        urls.append(image)
    elif isinstance(image, list):
        urls.extend(str(i) for i in image if isinstance(i, str))

    # Normalize and deduplicate
    normalized: list[str] = []
    seen: set[str] = set()
    for url in urls:
        url = url.strip()
        if url.startswith("//"):
            url = "https:" + url
        if url and url not in seen:
            seen.add(url)
            normalized.append(url)

    if normalized:
        fields["image_urls"] = normalized


def _extract_video_from_obj(obj: dict, fields: dict) -> None:
    if "video_url" in fields:
        return
    # Pattern: var_video.file.url
    var_video = obj.get("var_video")
    if isinstance(var_video, dict):
        f = var_video.get("file")
        if isinstance(f, dict) and f.get("url"):
            fields["video_url"] = f["url"]


def _extract_colors(obj: dict, fields: dict) -> None:
    if "colors" in fields and fields["colors"]:
        return
    colors: list[str] = []

    # Pattern: questions with type COLOR
    questions = obj.get("questions")
    if isinstance(questions, list):
        for q in questions:
            if isinstance(q, dict) and str(q.get("type", "")).upper() == "COLOR":
                answers = q.get("answers", [])
                if isinstance(answers, list):
                    for a in answers:
                        if isinstance(a, dict) and a.get("title"):
                            colors.append(a["title"])

    # Pattern: colorDescription string
    cd = obj.get("colorDescription")
    if isinstance(cd, str) and cd and not colors:
        colors = [cd]

    # Pattern: relatedVariantProducts with color_swatch.name or variantName
    related = obj.get("relatedVariantProducts")
    if isinstance(related, list) and not colors:
        for r in related:
            if isinstance(r, dict):
                # Try color_swatch.name first, then variantName
                swatch = r.get("color_swatch")
                vn = (swatch.get("name") if isinstance(swatch, dict) else None) or r.get("variantName")
                if vn and isinstance(vn, str) and vn not in colors:
                    colors.append(vn)
        # Also add current variant name
        current_vn = obj.get("variantName")
        if current_vn and isinstance(current_vn, str) and current_vn not in colors:
            colors.insert(0, current_vn)

    # Pattern: color_swatch or color_group
    if not colors:
        swatch = obj.get("color_swatch") or obj.get("color_group")
        if isinstance(swatch, dict) and swatch.get("name"):
            colors = [swatch["name"]]

    if colors:
        fields["colors"] = colors


def _extract_colors_from_all_data(embedded_json: dict[str, Any], fields: dict) -> None:
    """Extract colors from data structures not directly on the product object.

    Some sites store colorway info at a sibling level to the product object.
    Overrides existing colors if this source has more.
    """
    existing_count = len(fields.get("colors", []))

    # Look for colorwayImages at any nesting level
    for data in embedded_json.values():
        cw_images = _find_key_recursive(data, "colorwayImages")
        if isinstance(cw_images, list) and cw_images:
            colors = []
            for cw in cw_images:
                if isinstance(cw, dict):
                    desc = cw.get("colorDescription")
                    if desc and isinstance(desc, str) and desc not in colors:
                        colors.append(desc)
            if colors and len(colors) > existing_count:
                fields["colors"] = colors
                return


def _find_key_recursive(data: Any, key: str, depth: int = 0) -> Any:
    """Find a key's value anywhere in a nested structure."""
    if depth > 8:
        return None
    if isinstance(data, dict):
        if key in data:
            return data[key]
        for v in data.values():
            result = _find_key_recursive(v, key, depth + 1)
            if result is not None:
                return result
    elif isinstance(data, list):
        for item in data:
            result = _find_key_recursive(item, key, depth + 1)
            if result is not None:
                return result
    return None


def _extract_variants(obj: dict, fields: dict) -> None:
    if "variants" in fields and fields["variants"]:
        return
    variants: list[Variant] = []

    # Pattern: questions + skus cross-reference
    # Check this FIRST because it's richer than items when both exist
    questions = obj.get("questions")
    skus_list = obj.get("skus")
    if isinstance(questions, list) and isinstance(skus_list, list) and questions and skus_list:
        variants = _build_variants_from_questions(questions, skus_list)
        if variants:
            fields["variants"] = variants
            return

    # Pattern: items with name/sku/ean
    items = obj.get("items")
    if isinstance(items, list) and items:
        for item in items:
            if not isinstance(item, dict):
                continue
            # Skip items without identifiers (likely category labels, not variants)
            if not item.get("sku") and not item.get("ean"):
                continue
            attrs = {}
            name = item.get("name")
            if name:
                attrs["size"] = str(name)
            sku = item.get("sku") or item.get("item")
            ean = item.get("ean") or item.get("upc")
            # Check stock for availability
            stock = item.get("stock")
            available = True
            if isinstance(stock, (int, float)):
                available = stock > 0
            elif isinstance(stock, dict):
                total = sum(v for v in stock.values() if isinstance(v, (int, float)))
                available = total > 0
            variants.append(Variant(
                attributes=attrs,
                sku=str(sku) if sku else None,
                gtin=str(ean) if ean else None,
                available=available,
            ))
        if variants:
            fields["variants"] = variants
            return

    # Pattern: sizes list with label/gtins/status
    sizes = obj.get("sizes")
    if isinstance(sizes, list) and sizes and isinstance(sizes[0], dict) and "label" in sizes[0]:
        color_desc = obj.get("colorDescription", "")
        for size in sizes:
            if not isinstance(size, dict):
                continue
            attrs: dict[str, str] = {"size": str(size.get("label", ""))}
            if color_desc:
                attrs["color"] = color_desc
            gtin = None
            gtins = size.get("gtins")
            if isinstance(gtins, list) and gtins:
                first_gtin = gtins[0]
                if isinstance(first_gtin, dict):
                    gtin = first_gtin.get("gtin")
                elif isinstance(first_gtin, str):
                    gtin = first_gtin
            variants.append(Variant(
                attributes=attrs,
                sku=size.get("merchSkuId"),
                gtin=str(gtin) if gtin else None,
                available=size.get("status") == "ACTIVE",
            ))
        if variants:
            fields["variants"] = variants
            return


def _build_variants_from_questions(questions: list, skus_list: list) -> list[Variant]:
    """Build variants from questions/answers + SKU cross-reference."""
    # Build reverse lookup: sku_id -> {dimension_type: answer_title}
    sku_attrs: dict[str, dict[str, str]] = {}

    for q in questions:
        if not isinstance(q, dict):
            continue
        q_type = str(q.get("type", "")).lower()
        # Map question type to attribute name
        attr_name = q_type  # "color", "size", "item" (fit)
        if attr_name == "item":
            attr_name = "fit"

        answers = q.get("answers", [])
        if not isinstance(answers, list):
            continue
        for answer in answers:
            if not isinstance(answer, dict):
                continue
            title = answer.get("title", "")
            sku_ids = answer.get("skus", [])
            if not isinstance(sku_ids, list):
                continue
            for sid in sku_ids:
                sid_str = str(sid)
                if sid_str not in sku_attrs:
                    sku_attrs[sid_str] = {}
                sku_attrs[sid_str][attr_name] = title

    # Build variants from SKU list
    variants: list[Variant] = []
    for sku in skus_list:
        if not isinstance(sku, dict):
            continue
        sku_id = str(sku.get("id", ""))
        attrs = sku_attrs.get(sku_id, {})
        if not attrs:
            continue

        availability = sku.get("availability", {})
        available = True
        if isinstance(availability, dict):
            status = str(availability.get("status", "")).upper()
            available = status not in ("OUT_OF_STOCK", "UNAVAILABLE")

        variants.append(Variant(
            attributes=attrs,
            sku=sku_id,
            available=available,
        ))

    return variants


def _extract_category_hint(obj: dict, fields: dict) -> None:
    """Extract category if available in structured data (e.g. google_merchant_category)."""
    if "category" in fields:
        return
    gmc = obj.get("google_merchant_category")
    if isinstance(gmc, str) and gmc in VALID_CATEGORIES:
        fields["category"] = gmc
        return
    # Store raw category hints for later fuzzy matching (even if not exact taxonomy match)
    cat = obj.get("category")
    if isinstance(cat, str) and cat.strip():
        fields.setdefault("_category_hints", []).append(cat.strip())


# ----- JSON-LD Field Extractor -----

def _extract_fields_from_json_ld(ld: dict, fields: dict) -> None:
    """Extract Product fields from a JSON-LD block."""
    ld_type = ld.get("@type", "")

    # Skip non-product types
    if ld_type not in ("Product", "ProductGroup"):
        return

    if "name" not in fields and ld.get("name"):
        fields["name"] = html_lib.unescape(str(ld["name"]))

    if "brand" not in fields:
        brand = ld.get("brand")
        if isinstance(brand, dict):
            brand = brand.get("name")
        if brand:
            fields["brand"] = str(brand)

    if "description" not in fields and ld.get("description"):
        fields["description"] = ld["description"]

    # Price from offers
    if "price" not in fields:
        offers = ld.get("offers")
        if isinstance(offers, dict) and "price" in offers:
            fields["price"] = Price(
                price=float(offers["price"]),
                currency=offers.get("priceCurrency", "USD"),
            )

    # Images
    if "image_urls" not in fields or not fields.get("image_urls"):
        urls: list[str] = []
        image = ld.get("image")
        if isinstance(image, str):
            urls.append(image)
        elif isinstance(image, list):
            urls.extend(str(i) for i in image if isinstance(i, str))
        images_list = ld.get("images")
        if isinstance(images_list, list):
            urls.extend(str(i) for i in images_list if isinstance(i, str))
        # Normalize
        normalized = []
        for u in urls:
            u = u.strip()
            if u.startswith("//"):
                u = "https:" + u
            normalized.append(u)
        if normalized:
            fields["image_urls"] = normalized

    # Features from positiveNotes
    if "key_features" not in fields or not fields.get("key_features"):
        notes = ld.get("positiveNotes")
        if isinstance(notes, list) and notes:
            fields["key_features"] = [str(n) for n in notes if n]

    # Category from JSON-LD
    if "category" not in fields:
        cat = ld.get("category")
        if isinstance(cat, str) and cat.strip():
            if cat in VALID_CATEGORIES:
                fields["category"] = cat
            else:
                fields.setdefault("_category_hints", []).append(cat.strip())

    # Variants from hasVariant (ProductGroup)
    if "variants" not in fields or not fields.get("variants"):
        has_variant = ld.get("hasVariant", [])
        if isinstance(has_variant, list) and has_variant:
            variants: list[Variant] = []
            for v in has_variant:
                if not isinstance(v, dict):
                    continue
                attrs: dict[str, str] = {}
                if v.get("size"):
                    attrs["size"] = str(v["size"])
                if v.get("color"):
                    attrs["color"] = str(v["color"])
                if not attrs:
                    continue
                v_price = None
                v_offers = v.get("offers")
                if isinstance(v_offers, dict) and "price" in v_offers:
                    v_price = Price(
                        price=float(v_offers["price"]),
                        currency=v_offers.get("priceCurrency", "USD"),
                    )
                variants.append(Variant(
                    attributes=attrs,
                    sku=v.get("mpn"),
                    gtin=v.get("gtin"),
                    price=v_price,
                ))
            if variants:
                fields["variants"] = variants


# ----- OG Tag Extractor -----

def _extract_fields_from_og(og: dict, fields: dict) -> None:
    """Extract Product fields from Open Graph meta tags."""
    if "name" not in fields and og.get("title"):
        fields["name"] = og["title"]
    if "description" not in fields and og.get("description"):
        fields["description"] = og["description"]
    if ("image_urls" not in fields or not fields.get("image_urls")) and og.get("image"):
        fields["image_urls"] = [og["image"]]
    if "brand" not in fields and og.get("site_name"):
        fields["brand"] = og["site_name"]


# ----- Body Text Price Fallback -----

def _extract_price_from_text(text: str, fields: dict) -> None:
    """Extract price from visible body text via regex (fallback when not in structured data)."""
    match = re.search(r"\$(\d+(?:,\d{3})*(?:\.\d{2})?)", text)
    if match:
        amount = float(match.group(1).replace(",", ""))
        fields["price"] = Price(price=amount, currency="USD")


# =====================================================================
# Stage B: LLM Gap-Fill
# =====================================================================

async def _llm_fill_gaps(fields: dict, parsed: ParsedPage) -> tuple[dict, int, bool]:
    """Call LLM only for fields still missing after programmatic hydration.

    Returns (fields, llm_call_count, was_skipped).
    """
    missing: list[str] = []

    # Category always needs LLM unless we got an exact taxonomy match
    needs_category = "category" not in fields

    if "key_features" not in fields or not fields.get("key_features"):
        missing.append("key_features")
    if "colors" not in fields or not fields.get("colors"):
        missing.append("colors")
    if "description" not in fields or not fields.get("description"):
        missing.append("description")
    if "price" not in fields:
        missing.append("price")

    if not missing and not needs_category:
        logger.info("  All fields hydrated programmatically, skipping LLM")
        return fields, 0, True

    # Build compact LLM context
    fill_fields = (["category"] if needs_category else []) + missing
    context = _build_llm_context(fields, parsed, fill_fields)

    logger.info(f"  LLM filling: {fill_fields}")

    result = await _call_llm_for_gaps(context, fill_fields, fields, parsed)

    # Merge results
    if result.category and needs_category:
        fields["category"] = result.category
    if result.key_features and "key_features" in missing:
        fields["key_features"] = result.key_features
    if result.colors and "colors" in missing:
        fields["colors"] = result.colors
    if result.description and "description" in missing:
        fields["description"] = result.description
    if result.price and "price" in missing:
        currency = result.currency or "USD"
        fields["price"] = Price(price=result.price, currency=currency)

    return fields, 1, False


def _build_llm_context(fields: dict, parsed: ParsedPage, fill_fields: list[str]) -> str:
    """Build a compact summary for the LLM.

    When only category is needed, sends minimal context (~100 tokens).
    When other fields are needed, includes body text (~500+ tokens).
    """
    parts: list[str] = []

    if fields.get("name"):
        parts.append(f"Product Name: {fields['name']}")
    if fields.get("brand"):
        parts.append(f"Brand: {fields['brand']}")
    if fields.get("price"):
        p = fields["price"]
        if isinstance(p, Price):
            parts.append(f"Price: {p.price} {p.currency}")
        elif isinstance(p, dict):
            parts.append(f"Price: {p.get('price')} {p.get('currency', 'USD')}")
    if fields.get("description"):
        desc = str(fields["description"])[:500]
        parts.append(f"Description: {desc}")
    if fields.get("key_features"):
        features = fields["key_features"][:5]
        parts.append(f"Features: {'; '.join(str(f) for f in features)}")

    # Include category hints from structured data
    hints = fields.get("_category_hints", [])
    if hints:
        parts.append(f"Category hints from page: {'; '.join(hints)}")

    # Include breadcrumbs if available
    if parsed.breadcrumbs:
        parts.append(f"Breadcrumbs: {' > '.join(parsed.breadcrumbs)}")

    # Only include body text when we need fields beyond just category
    non_category_fields = [f for f in fill_fields if f != "category"]
    if non_category_fields and parsed.body_text:
        parts.append(f"\nPage Text:\n{parsed.body_text[:2000]}")

    return "\n".join(parts)


async def _call_llm_for_gaps(context: str, fill_fields: list[str], fields: dict, parsed: ParsedPage) -> LLMGapFill:
    """Call LLM to fill missing product fields."""
    system = (
        "You are a product data extraction assistant. "
        "Extract the requested fields from the product context provided. "
    )

    # If category is needed, include focused taxonomy subset
    if "category" in fill_fields:
        subset = _build_taxonomy_subset(fields, parsed)
        logger.info(f"  Taxonomy subset: {len(subset)} categories")
        system += (
            "For category, you MUST pick one EXACTLY from the provided list. "
            "IMPORTANT: Always choose the MOST SPECIFIC (deepest) category that fits. "
            "Never pick a broad top-level category like 'Hardware' or 'Home & Garden' "
            "when a more specific subcategory exists in the list."
        )
        category_block = f"\n\nValid categories (pick the MOST SPECIFIC one, copy it EXACTLY):\n" + "\n".join(subset)
    else:
        category_block = ""

    user = (
        f"Fill these missing fields: {', '.join(fill_fields)}\n\n"
        f"{context}"
        f"{category_block}"
    )

    result = await ai.responses(
        model=LLM_MODEL,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        text_format=LLMGapFill,
    )
    return result


# =====================================================================
# Stage C: Validation + Category Retry
# =====================================================================

async def _validate_product(fields: dict, parsed: ParsedPage) -> tuple[Product, bool, bool, bool, bool]:
    """Validate fields into a Product, trying fuzzy match before LLM retry.

    Returns (product, category_first_attempt_valid, category_fuzzy_matched,
             category_retry_needed, category_retry_succeeded).
    """
    _set_defaults(fields)

    # Try fuzzy matching the category BEFORE Pydantic validation
    cat_str = fields.get("category", "")
    if isinstance(cat_str, Category):
        cat_str = cat_str.name
    original_cat = cat_str
    fuzzy_resolved = False
    if isinstance(cat_str, str) and cat_str not in VALID_CATEGORIES:
        # Also try category hints from structured data
        candidates_to_try = [cat_str]
        for hint in fields.get("_category_hints", []):
            if hint not in candidates_to_try:
                candidates_to_try.append(hint)
        for candidate in candidates_to_try:
            fuzzy = _fuzzy_match_category(candidate)
            if fuzzy:
                logger.info(f"  Fuzzy matched category: '{candidate}' -> '{fuzzy}'")
                cat_str = fuzzy
                fuzzy_resolved = True
                break

    fields["category"] = {"name": cat_str}

    # Ensure price is a dict for Pydantic
    if isinstance(fields.get("price"), Price):
        fields["price"] = fields["price"].model_dump()

    # Ensure variants are dicts
    if "variants" in fields:
        fields["variants"] = [
            v.model_dump() if isinstance(v, Variant) else v
            for v in fields["variants"]
        ]

    # Save and remove internal hint fields before validation
    saved_hints = fields.pop("_category_hints", [])

    try:
        product = Product(**fields)
        return product, not fuzzy_resolved, fuzzy_resolved, False, False
    except ValidationError as e:
        # Check if it's a category validation error
        cat_errors = [err for err in e.errors() if "category" in str(err.get("loc", []))]
        if cat_errors:
            logger.info(f"  Category validation failed, retrying with taxonomy subset...")
            # Restore hints so _build_taxonomy_subset can use them during retry
            fields["_category_hints"] = saved_hints
            new_cat = await _retry_category(fields, parsed)
            fields.pop("_category_hints", None)
            if new_cat:
                fields["category"] = {"name": new_cat}
                product = Product(**fields)
                return product, False, False, True, True  # retry succeeded
            return Product(**fields), False, False, True, False  # retry failed (will raise)
        raise


async def _retry_category(fields: dict, parsed: ParsedPage) -> str | None:
    """Retry category selection with a broader taxonomy subset."""
    subset = _build_taxonomy_subset(fields, parsed, broaden=True)

    logger.info(f"  Retrying with {len(subset)} categories (broadened)")

    name = fields.get("name", "Unknown")
    brand = fields.get("brand", "Unknown")
    desc = str(fields.get("description", ""))[:300]

    prompt = (
        f"Choose the EXACT category for this product from the list below.\n\n"
        f"Product: {name}\nBrand: {brand}\nDescription: {desc}\n\n"
        f"Categories (pick one EXACTLY as written):\n"
        + "\n".join(subset[:300])  # Cap at 300 to stay within token budget
    )

    result = await ai.responses(
        model=LLM_MODEL,
        input=[{"role": "user", "content": prompt}],
        text_format=LLMGapFill,
    )

    if result.category and result.category in VALID_CATEGORIES:
        return result.category

    logger.warning(f"  Category retry also failed: '{result.category}'")
    return None


def _find_taxonomy_prefix(text: str) -> str:
    """Find the deepest matching taxonomy prefix from product text signals."""
    for prefix, keywords in _TAXONOMY_PREFIX_MAP:
        if any(kw in text for kw in keywords):
            return prefix
    return ""


def _build_taxonomy_subset(fields: dict, parsed: ParsedPage, broaden: bool = False) -> list[str]:
    """Build a focused taxonomy subset from product signals.

    Narrows the ~5500 taxonomy entries to a relevant subtree (often 6-50 entries)
    using deep keyword matching against product name, hints, and breadcrumbs.

    Args:
        broaden: If True, use one level above the focused prefix (for retry).
    """
    # Collect all text signals
    parts = [
        fields.get("name", ""),
        str(fields.get("description", ""))[:300],
        fields.get("brand", ""),
    ]
    for hint in fields.get("_category_hints", []):
        parts.append(hint)
    for bc in parsed.breadcrumbs:
        parts.append(bc)
    text = " ".join(parts).lower()

    prefix = _find_taxonomy_prefix(text)

    # For retry, go one level broader
    if broaden and prefix and " > " in prefix:
        prefix = prefix.rsplit(" > ", 1)[0]

    if not prefix:
        return sorted(VALID_CATEGORIES)[:200]

    subset = sorted(c for c in VALID_CATEGORIES if c.startswith(prefix))

    # If subset is too small, broaden one level at a time
    while len(subset) < 5 and " > " in prefix:
        prefix = prefix.rsplit(" > ", 1)[0]
        subset = sorted(c for c in VALID_CATEGORIES if c.startswith(prefix))

    if len(subset) < 5:
        return sorted(VALID_CATEGORIES)[:200]

    return subset[:250]


def _set_defaults(fields: dict) -> None:
    """Set default values for optional/list fields."""
    fields.setdefault("key_features", [])
    fields.setdefault("colors", [])
    fields.setdefault("variants", [])
    fields.setdefault("video_url", None)
    fields.setdefault("image_urls", [])
    fields.setdefault("description", "")
