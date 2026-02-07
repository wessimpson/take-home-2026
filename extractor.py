"""
Product data extractor: programmatic hydration + LLM gap-fill.

Three stages:
  A) Walk parsed structured data to fill Product fields (free)
  B) Call LLM only for missing fields — primarily category (cheap)
  C) Validate with Pydantic, retry category with taxonomy subset on failure
"""

import contextlib
import html as html_lib
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ValidationError

import ai
import taxonomy
from models import VALID_CATEGORIES, Category, Price, Product, Variant
from parser import ParsedPage

logger = logging.getLogger(__name__)

# Keys that signal "this dict is about a product" — generic e-commerce vocabulary only
PRODUCT_SIGNAL_KEYS = {
    "name",
    "title",
    "price",
    "prices",
    "offers",
    "description",
    "brand",
    "brandName",
    "sku",
    "image",
    "images",
    "media",
    "variants",
    "items",
    "skus",
    "sizes",
    "hasVariant",
    "google_merchant_category",
    "color",
    "colors",
    "questions",
}

# Model to use for LLM gap-filling
LLM_MODEL = "google/gemini-2.0-flash-lite-001"


# ===== Metrics =====

PRODUCT_FIELDS = [
    "name",
    "brand",
    "price",
    "description",
    "key_features",
    "image_urls",
    "video_url",
    "category",
    "colors",
    "variants",
]


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
        if val is not None and val != "" and val != [] and f not in metrics.fields_from_parser:
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
        metrics.category_resolution = "taxonomy_resolved"
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
# Generic Utility Functions
# =====================================================================

# Semantic key patterns for recursive extraction
_NAME_KEY_RE = re.compile(r"(?i)(^name$|^title$|productname|producttitle|fulltitle)")
_DESC_KEY_RE = re.compile(r"(?i)(description|desc$|summary|overview|excerpt)")
_FEATURE_KEY_RE = re.compile(
    r"(?i)(feature|benefit|highlight|specification|specs?$|bullet|bullet_point|^note|selling_point|usp)"
)
_COLOR_KEY_RE = re.compile(r"(?i)(color|colour|shade|swatch|hue|^finish$)")
_COLOR_NOISE_RE = re.compile(r"^(#[0-9a-fA-F]{3,8}|[A-Z]{2}\d{4}-\d{3}|rgba?\(|hsla?\()$")
_PRICE_KEY_RE = re.compile(
    r"(?i)^(price|amount|cost|currentPrice|salePrice|regularPrice|unitPrice|productPrice|basePrice|variantPrice)$"
)
_CURRENCY_KEY_RE = re.compile(r"(?i)^(currency|priceCurrency|currencyCode)$")
_COMPARE_PRICE_KEY_RE = re.compile(r"(?i)(compare|original|msrp|was|fullPrice|retail|initial|before|listPrice)")
_SKIP_IMAGE_RE = re.compile(r"(?i)(favicon|logo|pixel|tracking|analytics|1x1|spacer|icon\b|\.svg)")
_VIDEO_URL_RE = re.compile(r"https?://[^\s\"'<>]+\.(?:mp4|webm|m3u8)", re.IGNORECASE)

# Variant detection
_VARIANT_SKU_KEYS = re.compile(r"(?i)^(sku|id|item|code|mpn)$")
_VARIANT_GTIN_KEYS = re.compile(r"(?i)^(gtin|ean|upc|barcode|isbn)$")
_VARIANT_SIZE_KEYS = re.compile(r"(?i)^(size|label|dimension)$")
_VARIANT_COLOR_KEYS = re.compile(r"(?i)^(color|colour)$")
_VARIANT_AVAIL_KEYS = re.compile(r"(?i)^(status|available|availability|stock|instock)$")
_VARIANT_SIGNAL_KEYS = {
    "size",
    "sku",
    "ean",
    "upc",
    "gtin",
    "barcode",
    "label",
    "stock",
    "availability",
    "available",
    "price",
    "amount",
    "color",
    "colour",
    "title",
    "name",
    "option",
    "variant",
    "option1",
    "option2",
}


def _find_values_by_key_pattern(
    data: Any,
    key_patterns: list[re.Pattern],
    depth: int = 0,
    max_depth: int = 8,
) -> list[Any]:
    """Find all values in a nested structure whose keys match any of the given patterns."""
    if depth > max_depth:
        return []
    results = []
    if isinstance(data, dict):
        for k, v in data.items():
            if any(p.search(k) for p in key_patterns):
                results.append(v)
            results.extend(_find_values_by_key_pattern(v, key_patterns, depth + 1, max_depth))
    elif isinstance(data, list):
        for item in data:
            results.extend(_find_values_by_key_pattern(item, key_patterns, depth + 1, max_depth))
    return results


def _collect_image_urls_recursive(
    data: Any,
    depth: int = 0,
    max_depth: int = 10,
) -> list[str]:
    """Recursively collect all string values that look like image URLs."""
    if depth > max_depth:
        return []
    urls = []
    if isinstance(data, str):
        if _looks_like_image_url_string(data):
            urls.append(data)
    elif isinstance(data, dict):
        for v in data.values():
            urls.extend(_collect_image_urls_recursive(v, depth + 1, max_depth))
    elif isinstance(data, list):
        for item in data:
            urls.extend(_collect_image_urls_recursive(item, depth + 1, max_depth))
    return urls


def _looks_like_image_url_string(s: str) -> bool:
    """Check if a string looks like a product image URL."""
    s = s.strip()
    if not s.startswith(("http://", "https://", "//")):
        return False
    if _SKIP_IMAGE_RE.search(s):
        return False
    return bool(re.search(r"\.(jpg|jpeg|png|webp|gif|avif)", s, re.IGNORECASE))


def _collect_video_urls_recursive(
    data: Any,
    depth: int = 0,
    max_depth: int = 10,
) -> list[str]:
    """Recursively collect all string values that look like video URLs."""
    if depth > max_depth:
        return []
    urls = []
    if isinstance(data, str):
        if _VIDEO_URL_RE.search(data):
            urls.append(data)
    elif isinstance(data, dict):
        for v in data.values():
            urls.extend(_collect_video_urls_recursive(v, depth + 1, max_depth))
    elif isinstance(data, list):
        for item in data:
            urls.extend(_collect_video_urls_recursive(item, depth + 1, max_depth))
    return urls


def _extract_price_recursive(data: Any, depth: int = 0) -> Price | None:
    """Walk nested data to find and construct a Price from any price-like dict."""
    if depth > 10:
        return None
    if isinstance(data, dict):
        price_val = None
        currency = "USD"
        compare_val = None

        for k, v in data.items():
            if _PRICE_KEY_RE.match(k):
                if isinstance(v, (int, float)):
                    price_val = float(v)
                elif isinstance(v, str):
                    with contextlib.suppress(ValueError):
                        price_val = float(v.replace(",", ""))
            elif _CURRENCY_KEY_RE.match(k) and isinstance(v, str):
                currency = v
            elif _COMPARE_PRICE_KEY_RE.search(k) and isinstance(v, (int, float)):
                compare_val = float(v)

        if price_val is not None and price_val > 0:
            # Compare-at must be higher than sale price to make sense as a discount
            if compare_val and (compare_val == price_val or compare_val < price_val):
                compare_val = None
            return Price(price=price_val, currency=currency, compare_at_price=compare_val)

        # Recurse into children
        for v in data.values():
            result = _extract_price_recursive(v, depth + 1)
            if result:
                return result
    elif isinstance(data, list):
        for item in data:
            result = _extract_price_recursive(item, depth + 1)
            if result:
                return result
    return None


def _find_variant_arrays(data: Any, depth: int = 0, max_depth: int = 8) -> list[list[dict]]:
    """Find arrays of dicts that look like variant/SKU lists."""
    if depth > max_depth:
        return []
    results = []
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list) and len(v) >= 2 and _looks_like_variant_array(v):
                results.append(v)
            results.extend(_find_variant_arrays(v, depth + 1, max_depth))
    elif isinstance(data, list):
        if len(data) >= 2 and _looks_like_variant_array(data):
            results.append(data)
        for item in data:
            results.extend(_find_variant_arrays(item, depth + 1, max_depth))
    return results


def _looks_like_variant_array(arr: list) -> bool:
    """Check if a list of dicts looks like a variant/SKU array."""
    if not arr or not isinstance(arr[0], dict):
        return False
    sample = arr[:3]
    for item in sample:
        if not isinstance(item, dict):
            return False
        signal_count = len(_VARIANT_SIGNAL_KEYS & {k.lower() for k in item})
        if signal_count < 2:
            return False
    return True


def _normalize_and_dedup_urls(urls: list[str]) -> list[str]:
    """Normalize URLs and remove duplicates while preserving order."""
    result: list[str] = []
    seen: set[str] = set()
    for url in urls:
        url = url.strip()
        if url.startswith("//"):
            url = "https:" + url
        if url and url not in seen and url.startswith("http"):
            seen.add(url)
            result.append(url)
    return result


def _is_color_name(val: str) -> bool:
    """Check if a string looks like a color name (not a hex code, URL, or SKU)."""
    val = val.strip()
    if not val or len(val) > 80:
        return False
    if val.startswith(("http", "#", "rgba", "hsla")):
        return False
    if _COLOR_NOISE_RE.match(val):
        return False
    # Skip purely numeric or very short values (likely codes)
    return not (val.isdigit() or len(val) < 2)


def _clean_html(text: str) -> str:
    """Strip HTML tags and normalize whitespace."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# =====================================================================
# Stage A: Programmatic Hydration
# =====================================================================


def _hydrate_fields(parsed: ParsedPage) -> dict:
    """Walk all data sources in priority order to fill Product fields."""
    fields: dict[str, Any] = {}

    # 1. Embedded JSON (richest source for most pages)
    for _source_name, data in parsed.embedded_json.items():
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

    # 5. Look for colors across all embedded JSON data
    _extract_colors_from_all_data(parsed.embedded_json, fields)

    # 5b. Broad image sweep from embedded JSON (catches __NEXT_DATA__, etc.)
    # Merges with any images already found, placing high-res JSON URLs first.
    for _source_name, data in parsed.embedded_json.items():
        broad_urls = _normalize_and_dedup_urls(_collect_image_urls_recursive(data))
        if len(broad_urls) > len(fields.get("image_urls", [])):
            existing = fields.get("image_urls", [])
            # Prepend broad sweep URLs, then append any existing URLs not already present
            merged = list(broad_urls)
            seen = set(merged)
            for url in existing:
                if url not in seen:
                    merged.append(url)
                    seen.add(url)
            fields["image_urls"] = merged

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

    # 7. Breadcrumbs as category hint (exclude last segment — usually the product name)
    if "category" not in fields and parsed.breadcrumbs:
        cat_crumbs = parsed.breadcrumbs[:-1] if len(parsed.breadcrumbs) > 1 else parsed.breadcrumbs
        fields.setdefault("_category_hints", []).append(" > ".join(cat_crumbs))

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
    # Try common top-level keys first
    name = obj.get("name") or obj.get("title")
    if name and isinstance(name, str):
        fields["name"] = html_lib.unescape(name.strip())
        return
    # Recursive search for any key matching product name patterns
    candidates = _find_values_by_key_pattern(obj, [_NAME_KEY_RE])
    for c in candidates:
        if isinstance(c, str) and len(c.strip()) > 5:
            fields["name"] = html_lib.unescape(c.strip())
            return


def _extract_brand(obj: dict, fields: dict) -> None:
    if "brand" in fields:
        return
    brand = obj.get("brandName") or obj.get("brand")
    if isinstance(brand, dict):
        brand = brand.get("name")
    if isinstance(brand, list):
        brand = brand[0] if brand else None
    if not brand or not isinstance(brand, str):
        # Recursive fallback for keys matching "brand" or "brandName"
        brand_pattern = re.compile(r"(?i)^brand(Name)?$")
        candidates = _find_values_by_key_pattern(obj, [brand_pattern])
        for c in candidates:
            if isinstance(c, str) and c.strip():
                brand = c.strip()
                break
            elif isinstance(c, dict) and c.get("name"):
                brand = c["name"]
                break
    if brand and isinstance(brand, str):
        fields["brand"] = brand.strip()


def _extract_price_field(obj: dict, fields: dict) -> None:
    if "price" in fields:
        return

    # JSON-LD offers pattern (schema.org standard)
    offers = obj.get("offers")
    if isinstance(offers, list) and offers:
        offers = offers[0]
    if isinstance(offers, dict) and "price" in offers:
        fields["price"] = Price(
            price=float(offers["price"]),
            currency=offers.get("priceCurrency", "USD"),
        )
        return

    # Generic recursive price extraction
    price = _extract_price_recursive(obj)
    if price:
        fields["price"] = price


def _extract_description(obj: dict, fields: dict) -> None:
    if "description" in fields:
        return
    # Try common top-level keys first
    desc = obj.get("description") or obj.get("excerpt")
    if desc and isinstance(desc, str) and len(desc.strip()) > 20:
        top_level_desc = _clean_html(desc)
    else:
        top_level_desc = ""
    # Recursive search for description-like keys, pick the longest
    candidates = _find_values_by_key_pattern(obj, [_DESC_KEY_RE])
    best = top_level_desc
    for c in candidates:
        if isinstance(c, str) and len(c.strip()) > len(best):
            best = _clean_html(c.strip())
    if len(best) > 60:
        fields["description"] = best


def _extract_features(obj: dict, fields: dict) -> None:
    if fields.get("key_features"):
        return
    features: list[str] = []

    # Schema.org positiveNotes (standard field)
    notes = obj.get("positiveNotes")
    if isinstance(notes, list):
        features.extend(str(n) for n in notes if n)

    # Recursive search for feature/benefit/highlight arrays
    if not features:
        candidates = _find_values_by_key_pattern(obj, [_FEATURE_KEY_RE])
        for candidate in candidates:
            if isinstance(candidate, list):
                for item in candidate:
                    if isinstance(item, str) and item.strip():
                        features.append(item.strip())
                    elif isinstance(item, dict):
                        # Extract text from dict items (e.g., {body: [...]} sections)
                        for v in item.values():
                            if isinstance(v, str) and len(v.strip()) > 5:
                                features.append(v.strip())
                            elif isinstance(v, list):
                                features.extend(str(x) for x in v if x and isinstance(x, str))
            elif isinstance(candidate, str) and len(candidate.strip()) > 5:
                features.append(candidate.strip())

    # Bullet-formatted description fallback
    if not features:
        desc = obj.get("description", "")
        if isinstance(desc, str) and "\n-" in desc:
            for line in desc.split("\n"):
                line = line.strip().lstrip("- ").strip()
                if line and len(line) > 3:
                    features.append(line)

    if features:
        # Deduplicate while preserving order
        seen = set()
        unique = []
        for f in features:
            if f not in seen:
                seen.add(f)
                unique.append(f)
        fields["key_features"] = unique


def _extract_images_from_obj(obj: dict, fields: dict) -> None:
    if fields.get("image_urls"):
        return
    urls: list[str] = []

    # Generic top-level keys
    image = obj.get("image")
    if isinstance(image, str):
        urls.append(image)
    elif isinstance(image, list):
        urls.extend(str(i) for i in image if isinstance(i, str))

    images = obj.get("images")
    if isinstance(images, list):
        for img in images:
            if isinstance(img, str):
                urls.append(img)

    media_list = obj.get("media")
    if isinstance(media_list, list):
        for m in media_list:
            if isinstance(m, dict) and m.get("src"):
                urls.append(m["src"])

    # Recursive collection from all nested data
    recursive_urls = _collect_image_urls_recursive(obj)
    urls.extend(recursive_urls)

    normalized = _normalize_and_dedup_urls(urls)
    if normalized:
        fields["image_urls"] = normalized


def _extract_video_from_obj(obj: dict, fields: dict) -> None:
    if "video_url" in fields:
        return
    # Recursive search for video URLs at any depth
    video_urls = _collect_video_urls_recursive(obj)
    if video_urls:
        fields["video_url"] = video_urls[0]


def _extract_colors(obj: dict, fields: dict) -> None:
    if fields.get("colors"):
        return
    colors: list[str] = []

    # Generic: questions with type COLOR (common quiz/dimension pattern)
    questions = obj.get("questions")
    if isinstance(questions, list):
        for q in questions:
            if isinstance(q, dict) and str(q.get("type", "")).upper() == "COLOR":
                answers = q.get("answers", [])
                if isinstance(answers, list):
                    for a in answers:
                        if isinstance(a, dict) and a.get("title"):
                            colors.append(a["title"])

    # Recursive search for color-like values
    if not colors:
        candidates = _find_values_by_key_pattern(obj, [_COLOR_KEY_RE])
        for c in candidates:
            if isinstance(c, str) and _is_color_name(c):
                if c.strip() not in colors:
                    colors.append(c.strip())
            elif isinstance(c, dict):
                name = c.get("name") or c.get("title")
                if name and isinstance(name, str) and _is_color_name(name) and name not in colors:
                    colors.append(name)
            elif isinstance(c, list):
                for item in c:
                    if isinstance(item, str) and _is_color_name(item):
                        if item.strip() not in colors:
                            colors.append(item.strip())
                    elif isinstance(item, dict):
                        name = item.get("name") or item.get("title")
                        if name and isinstance(name, str) and _is_color_name(name) and name not in colors:
                            colors.append(name)

    if colors:
        fields["colors"] = colors


def _extract_colors_from_all_data(embedded_json: dict[str, Any], fields: dict) -> None:
    """Extract colors from embedded JSON sources outside the main product object."""
    existing_count = len(fields.get("colors", []))

    for data in embedded_json.values():
        candidates = _find_values_by_key_pattern(data, [_COLOR_KEY_RE])
        colors = []
        for c in candidates:
            if isinstance(c, str) and _is_color_name(c):
                if c.strip() not in colors:
                    colors.append(c.strip())
            elif isinstance(c, list):
                for item in c:
                    if isinstance(item, dict):
                        # Extract color name from objects in color arrays
                        name = None
                        for k, v in item.items():
                            if _COLOR_KEY_RE.search(k) and isinstance(v, str) and _is_color_name(v):
                                name = v.strip()
                                break
                        if not name:
                            name = item.get("name") or item.get("title")
                        if name and isinstance(name, str) and _is_color_name(name) and name not in colors:
                            colors.append(name)
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
    if fields.get("variants"):
        return

    # Generic: questions + skus cross-reference (common quiz/dimension pattern)
    questions = obj.get("questions")
    skus_list = obj.get("skus")
    if isinstance(questions, list) and isinstance(skus_list, list) and questions and skus_list:
        variants = _build_variants_from_questions(questions, skus_list)
        if variants:
            fields["variants"] = variants
            return

    # Generic: items with name/sku/ean
    items = obj.get("items")
    if isinstance(items, list) and items:
        variants: list[Variant] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            if not item.get("sku") and not item.get("ean"):
                continue
            attrs = {}
            name = item.get("name")
            if name:
                attrs["size"] = str(name)
            sku = item.get("sku") or item.get("item")
            ean = item.get("ean") or item.get("upc")
            stock = item.get("stock")
            available = True
            if isinstance(stock, (int, float)):
                available = stock > 0
            elif isinstance(stock, dict):
                total = sum(v for v in stock.values() if isinstance(v, (int, float)))
                available = total > 0
            variants.append(
                Variant(
                    attributes=attrs,
                    sku=str(sku) if sku else None,
                    gtin=str(ean) if ean else None,
                    available=available,
                )
            )
        if variants:
            fields["variants"] = variants
            return

    # Generic variant array detection at any depth
    variant_arrays = _find_variant_arrays(obj)
    if variant_arrays:
        best_array = max(variant_arrays, key=len)
        variants = _build_variants_from_generic_array(best_array)
        if variants:
            fields["variants"] = variants


def _build_variants_from_generic_array(arr: list[dict]) -> list[Variant]:
    """Build Variant objects from a generic array of variant-like dicts."""
    variants: list[Variant] = []
    for item in arr:
        if not isinstance(item, dict):
            continue

        attrs: dict[str, str] = {}
        sku = None
        gtin = None
        available = True

        for k, v in item.items():
            if _VARIANT_SIZE_KEYS.match(k):
                attrs["size"] = str(v)
            elif _VARIANT_COLOR_KEYS.match(k):
                if isinstance(v, str):
                    attrs["color"] = v
                elif isinstance(v, dict) and v.get("name"):
                    attrs["color"] = v["name"]
            elif _VARIANT_SKU_KEYS.match(k) and sku is None:
                sku = str(v)
            elif _VARIANT_GTIN_KEYS.match(k):
                if isinstance(v, str):
                    gtin = v
                elif isinstance(v, list) and v:
                    first = v[0]
                    if isinstance(first, str):
                        gtin = first
                    elif isinstance(first, dict):
                        gtin = str(next(iter(first.values()), ""))
            elif _VARIANT_AVAIL_KEYS.match(k):
                if isinstance(v, str):
                    available = v.upper() not in ("OUT_OF_STOCK", "UNAVAILABLE", "INACTIVE", "SOLD_OUT")
                elif isinstance(v, bool):
                    available = v
                elif isinstance(v, (int, float)):
                    available = v > 0
                elif isinstance(v, dict):
                    status = str(v.get("status", "")).upper()
                    available = status not in ("OUT_OF_STOCK", "UNAVAILABLE")

        if not attrs:
            name = item.get("name")
            if name:
                attrs["size"] = str(name)

        if attrs or sku:
            variants.append(
                Variant(
                    attributes=attrs,
                    sku=sku,
                    gtin=gtin,
                    available=available,
                )
            )

    return variants


def _build_variants_from_questions(questions: list, skus_list: list) -> list[Variant]:
    """Build variants from questions/answers + SKU cross-reference."""
    sku_attrs: dict[str, dict[str, str]] = {}

    for q in questions:
        if not isinstance(q, dict):
            continue
        q_type = str(q.get("type", "")).lower()
        attr_name = q_type

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

        variants.append(
            Variant(
                attributes=attrs,
                sku=sku_id,
                available=available,
            )
        )

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
        desc_text = str(ld["description"]).strip()
        if len(desc_text) > 60:
            fields["description"] = desc_text

    # Price from offers
    if "price" not in fields:
        offers = ld.get("offers")
        if isinstance(offers, list) and offers:
            offers = offers[0]
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
                if isinstance(v_offers, list) and v_offers:
                    v_offers = v_offers[0]
                if isinstance(v_offers, dict) and "price" in v_offers:
                    v_price = Price(
                        price=float(v_offers["price"]),
                        currency=v_offers.get("priceCurrency", "USD"),
                    )
                variants.append(
                    Variant(
                        attributes=attrs,
                        sku=v.get("mpn"),
                        gtin=v.get("gtin"),
                        price=v_price,
                    )
                )
            if variants:
                fields["variants"] = variants


# ----- OG Tag Extractor -----


def _extract_fields_from_og(og: dict, fields: dict) -> None:
    """Extract Product fields from Open Graph meta tags."""
    if "name" not in fields and og.get("title"):
        fields["name"] = og["title"]
    if "description" not in fields and og.get("description"):
        og_desc = str(og["description"]).strip()
        if len(og_desc) > 60:
            fields["description"] = og_desc
    if ("image_urls" not in fields or not fields.get("image_urls")) and og.get("image"):
        fields["image_urls"] = [og["image"]]
    if "brand" not in fields and og.get("site_name"):
        fields["brand"] = og["site_name"]

    # Price from og:price:amount or product:price:amount meta tags
    if "price" not in fields:
        price_str = og.get("price:amount")
        if price_str:
            try:
                price_val = float(price_str)
                currency = og.get("price:currency", "USD")
                if price_val > 0:
                    fields["price"] = Price(price=price_val, currency=currency)
            except (ValueError, TypeError):
                pass


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
        "You are a product data extraction assistant. Extract the requested fields from the product context provided. "
    )

    # If category is needed, use taxonomy tree to narrow candidates
    if "category" in fill_fields:
        signals = _collect_taxonomy_signals(fields, parsed)
        confident, candidates = taxonomy.classify(signals, top_n=15)
        if confident:
            # Taxonomy tree found a confident match — skip LLM for category
            logger.info(f"  Taxonomy tree match: {confident}")
            fields["category"] = confident
            fill_fields = [f for f in fill_fields if f != "category"]
            if not fill_fields:
                return LLMGapFill(category=confident)
        subset = candidates if len(candidates) >= 3 else sorted(VALID_CATEGORIES)[:200]
        logger.info(f"  Taxonomy candidates: {len(subset)} categories")
        system += (
            "For category, you MUST pick one EXACTLY from the provided list. "
            "IMPORTANT: Always choose the MOST SPECIFIC (deepest) category that fits. "
            "Never pick a broad top-level category like 'Hardware' or 'Home & Garden' "
            "when a more specific subcategory exists in the list."
        )
        category_block = "\n\nValid categories (pick the MOST SPECIFIC one, copy it EXACTLY):\n" + "\n".join(subset)
    else:
        category_block = ""

    user = f"Fill these missing fields: {', '.join(fill_fields)}\n\n{context}{category_block}"

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

    # Try resolving the category via taxonomy tree BEFORE Pydantic validation
    cat_str = fields.get("category", "")
    if isinstance(cat_str, Category):
        cat_str = cat_str.name
    fuzzy_resolved = False
    if isinstance(cat_str, str) and cat_str not in VALID_CATEGORIES:
        # Try the category string and any hints from structured data
        candidates_to_try = [cat_str]
        for hint in fields.get("_category_hints", []):
            if hint not in candidates_to_try:
                candidates_to_try.append(hint)
        for candidate in candidates_to_try:
            resolved = taxonomy.resolve(candidate)
            if resolved:
                logger.info(f"  Taxonomy resolved: '{candidate}' -> '{resolved}'")
                cat_str = resolved
                fuzzy_resolved = True
                break

        # If resolve() failed, try classify() with all available signals
        if not fuzzy_resolved:
            classify_signals = list(candidates_to_try)
            if fields.get("name"):
                classify_signals.append(fields["name"])
            if fields.get("description"):
                classify_signals.append(str(fields["description"])[:200])
            confident, _ = taxonomy.classify(classify_signals)
            if confident:
                logger.info(f"  Taxonomy classified: '{cat_str}' -> '{confident}'")
                cat_str = confident
                fuzzy_resolved = True

    fields["category"] = {"name": cat_str}

    # Ensure price is a dict for Pydantic
    if isinstance(fields.get("price"), Price):
        fields["price"] = fields["price"].model_dump()

    # Ensure variants are dicts
    if "variants" in fields:
        fields["variants"] = [v.model_dump() if isinstance(v, Variant) else v for v in fields["variants"]]

    # Save and remove internal hint fields before validation
    saved_hints = fields.pop("_category_hints", [])

    try:
        product = Product(**fields)
        return product, not fuzzy_resolved, fuzzy_resolved, False, False
    except ValidationError as e:
        # Check if it's a category validation error
        cat_errors = [err for err in e.errors() if "category" in str(err.get("loc", []))]
        if cat_errors:
            logger.info("  Category validation failed, retrying with taxonomy subset...")
            # Restore hints so taxonomy signals can use them during retry
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
    # Use taxonomy tree: get candidates and broaden from the best match
    signals = _collect_taxonomy_signals(fields, parsed)
    _, candidates = taxonomy.classify(signals, top_n=15)
    if candidates:
        # Broaden from the top candidate's subtree
        subset = taxonomy.broaden(candidates[0])
    else:
        subset = sorted(VALID_CATEGORIES)[:200]

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


def _collect_taxonomy_signals(fields: dict, parsed: ParsedPage) -> list[str]:
    """Collect product text signals for taxonomy classification.

    Excludes brand (brand names like 'L.L.Bean' cause false matches in taxonomy).
    Skips last breadcrumb segment (usually the product name, not a category).
    """
    signals = []
    if fields.get("name"):
        signals.append(fields["name"])
    if fields.get("description"):
        signals.append(str(fields["description"])[:300])
    for hint in fields.get("_category_hints", []):
        signals.append(hint)
    # Skip last breadcrumb (usually the product name, not a category level)
    bc_list = parsed.breadcrumbs[:-1] if len(parsed.breadcrumbs) > 1 else parsed.breadcrumbs
    for bc in bc_list:
        signals.append(bc)
    return signals


def _set_defaults(fields: dict) -> None:
    """Set default values for optional/list fields."""
    fields.setdefault("key_features", [])
    fields.setdefault("colors", [])
    fields.setdefault("variants", [])
    fields.setdefault("video_url", None)
    fields.setdefault("image_urls", [])
    fields.setdefault("description", "")
