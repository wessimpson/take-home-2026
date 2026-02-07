"""
Product data extractor: programmatic hydration + LLM gap-fill + QA validation.

Four stages:
  A) Walk parsed structured data to fill Product fields (free)
  B) Call LLM only for missing fields — primarily category (cheap)
  C) Validate with Pydantic, retry category with taxonomy subset on failure
  D) LLM QA validation — flag and correct discrepancies before output
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
from models import VALID_CATEGORIES, Category, Price, Product, Variant, VariantDimension
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
    "variant_dimensions",
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
    # Platform-aware extraction
    platform_detected: str = ""
    js_framework_detected: str = ""
    platform_extraction_used: bool = False
    dom_variants_used: bool = False
    dom_sale_price_used: bool = False
    breadcrumb_source: str = ""  # json_ld, microdata, html_nav, none
    # QA validation (Stage D)
    qa_time: float = 0.0
    qa_passed: bool = True
    qa_issues: list[str] = field(default_factory=list)
    qa_corrections_applied: int = 0


# ===== LLM Response Schema =====


class LLMGapFill(BaseModel):
    category: str | None = None
    key_features: list[str] | None = None
    colors: list[str] | None = None
    description: str | None = None
    price: float | None = None
    currency: str | None = None


class ImageVerdict(BaseModel):
    """Per-image assessment from the vision model."""

    index: int
    keep: bool
    color: str | None = None  # which color this image shows, if identifiable


class QAResult(BaseModel):
    """LLM QA validation output — vision-based."""

    passed: bool
    issues: list[str] = []
    image_verdicts: list[ImageVerdict] = []
    # Field corrections
    name: str | None = None
    brand: str | None = None
    category: str | None = None
    colors: list[str] | None = None
    key_features: list[str] | None = None
    has_duplicate_variants: bool = False


# QA validation toggle
QA_ENABLED = True


# ===== Main Entry Point =====


async def extract_product(parsed: ParsedPage, filename: str) -> tuple[Product, ExtractionMetrics]:
    """Full extraction pipeline: hydrate -> LLM fill -> validate. Returns product and metrics."""
    metrics = ExtractionMetrics(filename=filename)
    metrics.platform_detected = parsed.platform
    metrics.js_framework_detected = parsed.js_framework
    t_start = time.monotonic()

    # Stage A: Programmatic hydration (free)
    t0 = time.monotonic()
    fields = _hydrate_fields(parsed)
    metrics.hydrate_time = time.monotonic() - t0

    # Record platform-aware extraction flags
    metrics.platform_extraction_used = fields.pop("_platform_extraction_used", False)
    metrics.dom_variants_used = fields.pop("_dom_variants_used", False)
    metrics.dom_sale_price_used = fields.pop("_dom_sale_price_used", False)

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

    # Stage D: LLM QA validation (catch discrepancies)
    t0 = time.monotonic()
    product, qa_issues = await _qa_check(product, parsed)
    metrics.qa_time = time.monotonic() - t0
    metrics.qa_passed = len(qa_issues) == 0
    metrics.qa_issues = qa_issues
    if qa_issues:
        metrics.llm_calls += 1
        for issue in qa_issues:
            logger.info(f"  QA: {issue}")

    # Output richness (after QA corrections)
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
_SKIP_IMAGE_RE = re.compile(r"(?i)(favicon|logo|pixel|tracking|analytics|1x1|spacer|icon\b|\.svg|flyout)")
_VIDEO_URL_RE = re.compile(r"https?://[^\s\"'<>]+\.(?:mp4|webm|m3u8)", re.IGNORECASE)

# Variant detection — regex patterns for matching keys in arbitrary JSON structures
_VARIANT_SKU_KEYS = re.compile(r"(?i)^(sku|id|item|code|mpn|itemNumber|productCode)$")
_VARIANT_GTIN_KEYS = re.compile(r"(?i)^(gtin|gtin\d{0,2}|ean|upc|barcode|isbn)$")
_VARIANT_SIZE_KEYS = re.compile(r"(?i)^(size|sizing|shoe_?size|clothing_?size|apparel_?size|label|dimension|taille)$")
_VARIANT_COLOR_KEYS = re.compile(r"(?i)^(color|colour|colorway|shade|hue|color_?name|selectedColor|swatch)$")
_VARIANT_AVAIL_KEYS = re.compile(r"(?i)^(status|available|availability|stock|instock|in_stock|inventoryStatus)$")
_VARIANT_IMAGE_KEYS = re.compile(
    r"(?i)^(image|featured_image|image_url|imageUrl|img|photo|thumbnail)$"
)
_VARIANT_URL_KEYS = re.compile(r"(?i)^(url|href|link|product_?url|variant_?url|permalink)$")
# Additional dimension detection for generic arrays
_VARIANT_FIT_KEYS = re.compile(r"(?i)^(fit|cut|silhouette)$")
_VARIANT_WIDTH_KEYS = re.compile(r"(?i)^(width|shoe_?width)$")
_VARIANT_LENGTH_KEYS = re.compile(r"(?i)^(length|inseam|leg_?length)$")
_VARIANT_MATERIAL_KEYS = re.compile(r"(?i)^(material|fabric|composition|textile)$")
_VARIANT_STYLE_KEYS = re.compile(r"(?i)^(style|edition|model|version)$")
_VARIANT_PATTERN_KEYS = re.compile(r"(?i)^(pattern|print|motif|design)$")
_VARIANT_FLAVOR_KEYS = re.compile(r"(?i)^(flavor|flavour|scent|fragrance)$")
_VARIANT_STORAGE_KEYS = re.compile(r"(?i)^(storage|memory|capacity|ram)$")
_VARIANT_FINISH_KEYS = re.compile(r"(?i)^(finish|surface|coating)$")
_VARIANT_SIGNAL_KEYS = {
    "size",
    "sizing",
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
    "colorway",
    "title",
    "name",
    "option",
    "variant",
    "option1",
    "option2",
    "option3",
    "fit",
    "width",
    "length",
    "material",
    "style",
    "pattern",
    "flavor",
    "storage",
    "finish",
    "inseam",
    "capacity",
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


# Keys that indicate related/recommended product arrays (skip during image collection)
_RELATED_PRODUCTS_KEY_RE = re.compile(
    r"(?i)(related|recommend|similar|cross.?sell|also.?like|suggested|complementary|"
    r"frequently.?bought|upsell|you.?may|sponsored|recently.?viewed)"
)


def _collect_image_urls_recursive(
    data: Any,
    depth: int = 0,
    max_depth: int = 10,
    skip_related_keys: bool = False,
) -> list[str]:
    """Recursively collect all string values that look like image URLs.

    When skip_related_keys is True, skips dict entries whose key matches
    related/recommended product patterns to prevent cross-product contamination.
    """
    if depth > max_depth:
        return []
    urls = []
    if isinstance(data, str):
        if _looks_like_image_url_string(data):
            urls.append(data)
    elif isinstance(data, dict):
        for k, v in data.items():
            if skip_related_keys and isinstance(k, str) and _RELATED_PRODUCTS_KEY_RE.search(k):
                continue
            urls.extend(_collect_image_urls_recursive(v, depth + 1, max_depth, skip_related_keys))
    elif isinstance(data, list):
        for item in data:
            urls.extend(_collect_image_urls_recursive(item, depth + 1, max_depth, skip_related_keys))
    return urls


def _looks_like_image_url_string(s: str) -> bool:
    """Check if a string looks like a product image URL."""
    s = s.strip()
    if not s.startswith(("http://", "https://", "//")):
        return False
    if _SKIP_IMAGE_RE.search(s):
        return False
    return bool(re.search(r"\.(jpg|jpeg|png|webp|gif|avif)", s, re.IGNORECASE))


def _looks_like_media_src(s: str) -> bool:
    """Relaxed image URL check for media-lookup-resolved URLs.

    Media lookup tables are a trusted source — the page's own structured data
    mapping media IDs to src URLs. CDN image servers (e.g. Akamai) often serve
    images without file extensions, so we only check for valid HTTP(S) and
    skip obvious non-image patterns.
    """
    s = s.strip()
    if not s.startswith(("http://", "https://", "//")):
        return False
    return not _SKIP_IMAGE_RE.search(s)


def _extract_variant_image(value: Any) -> str | None:
    """Extract a single image URL from a variant's image field value."""
    if isinstance(value, str) and _looks_like_image_url_string(value):
        return ("https:" + value) if value.startswith("//") else value
    if isinstance(value, dict):
        src = value.get("src") or value.get("url")
        if isinstance(src, str) and _looks_like_image_url_string(src):
            return ("https:" + src) if src.startswith("//") else src
    if isinstance(value, list) and value:
        return _extract_variant_image(value[0])
    return None


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


def _match_variant_images_by_color(
    variants: list[Variant], image_urls: list[str]
) -> list[Variant]:
    """Fallback: match product-level images to color variants by URL substring."""
    if not image_urls:
        return variants

    updated = []
    for variant in variants:
        if variant.image_url is not None:
            updated.append(variant)
            continue

        color = variant.attributes.get("color", "").strip()
        if not color:
            updated.append(variant)
            continue

        # Normalize: "Jet Black" -> "jet-black" / "jet_black" / "jetblack"
        color_lower = color.lower()
        slugs = [
            re.sub(r"[^a-z0-9]+", "-", color_lower).strip("-"),
            re.sub(r"[^a-z0-9]+", "_", color_lower).strip("_"),
            re.sub(r"[^a-z0-9]", "", color_lower),
        ]

        match = None
        for url in image_urls:
            path = url.split("?")[0].lower()
            if any(s and s in path for s in slugs):
                match = url
                break

        if match:
            updated.append(variant.model_copy(update={"image_url": match}))
        else:
            updated.append(variant)

    return updated


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

    # 0. Platform-specific targeted extraction (highest priority)
    platform_obj = None
    if parsed.platform == "shopify":
        platform_obj = _extract_shopify_product(parsed)
    elif parsed.js_framework == "next.js":
        platform_obj = _extract_nextjs_product(parsed)

    if platform_obj:
        _extract_fields_from_object(platform_obj, fields)
        fields["_platform_extraction_used"] = True

    # 1. Embedded JSON (richest source for most pages — fills gaps from step 0)
    for _source_name, data in parsed.embedded_json.items():
        product_obj = _find_product_object(data)
        if product_obj:
            _extract_fields_from_object(product_obj, fields, data_root=data)

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
    # Uses skip_related_keys to avoid collecting images from related/recommended products.
    for _source_name, data in parsed.embedded_json.items():
        broad_urls = _normalize_and_dedup_urls(
            _collect_image_urls_recursive(data, skip_related_keys=True)
        )
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

    # 8. Match product-level images to color variants by URL heuristic
    if fields.get("variants") and fields.get("image_urls"):
        fields["variants"] = _match_variant_images_by_color(
            fields["variants"], fields["image_urls"]
        )

    # 8b. Resolve media IDs for variant images
    if fields.get("variants") and parsed.media_lookups:
        fields["variants"] = _resolve_variant_media_ids(
            fields["variants"], parsed.embedded_json, parsed.media_lookups
        )

    # 8c. Inject active dimension values into variants
    if fields.get("variants") and parsed.active_selections:
        fields["variants"] = _inject_active_selections(
            fields["variants"], parsed.active_selections
        )

    # 8d. Extract sibling variants and merge
    sibling_variants = _extract_sibling_variants(
        parsed.embedded_json, parsed.active_selections, parsed.page_url
    )
    if sibling_variants:
        existing = fields.get("variants", [])
        fields["variants"] = existing + sibling_variants

    # 9. DOM-based variant fallback (only when JSON variants not found)
    if not fields.get("variants") and parsed.dom_variants:
        variants = _build_variants_from_dom(parsed.dom_variants)
        if variants:
            fields["variants"] = variants
            fields["_dom_variants_used"] = True

    # 9b. DOM-based color fallback
    if not fields.get("colors") and parsed.dom_variants:
        for sig in parsed.dom_variants:
            if sig["type"] == "color" and sig["values"]:
                fields["colors"] = sig["values"]
                break

    # 10. DOM-based sale price enrichment
    if parsed.dom_sale_price.get("has_sale"):
        dom_sale = parsed.dom_sale_price
        if "price" in fields and isinstance(fields["price"], Price):
            if fields["price"].compare_at_price is None and dom_sale.get("original_price"):
                original = dom_sale["original_price"]
                if original > fields["price"].price:
                    fields["price"] = Price(
                        price=fields["price"].price,
                        currency=fields["price"].currency,
                        compare_at_price=original,
                    )
                    fields["_dom_sale_price_used"] = True
        elif "price" not in fields and dom_sale.get("sale_price") and dom_sale.get("original_price"):
            fields["price"] = Price(
                price=dom_sale["sale_price"],
                currency="USD",
                compare_at_price=dom_sale["original_price"],
            )
            fields["_dom_sale_price_used"] = True

    # 11. Derive variant_dimensions from variants if not already set (covers all extraction paths)
    if fields.get("variants") and not fields.get("variant_dimensions"):
        fields["variant_dimensions"] = _build_dimensions_from_variants(fields["variants"])

    return fields


# ----- Platform-Specific Extractors -----


def _extract_shopify_product(parsed: ParsedPage) -> dict | None:
    """Targeted extraction for Shopify pages.

    Shopify stores product data in ProductJson-* script tags or analytics tracking.
    """
    # Priority 1: ProductJson-* script tag (most complete source)
    for key, data in parsed.embedded_json.items():
        if key.startswith("ProductJson") and isinstance(data, dict):
            return data

    # Priority 2: Analytics tracking product
    tracked = parsed.embedded_json.get("_analytics_tracked_product")
    if isinstance(tracked, dict) and tracked.get("name"):
        return tracked

    return None


def _extract_nextjs_product(parsed: ParsedPage) -> dict | None:
    """Targeted extraction for Next.js pages.

    Navigate directly to __NEXT_DATA__.props.pageProps instead of generic walking.
    """
    next_data = parsed.embedded_json.get("__NEXT_DATA__")
    if not isinstance(next_data, dict):
        return None

    page_props = next_data.get("props", {}).get("pageProps", {})
    if not page_props:
        return None

    return _find_product_object(page_props)


def _build_variants_from_dom(dom_signals: list[dict]) -> list[Variant]:
    """Build Variant objects from DOM-detected size/color options.

    Lower fidelity than JSON-based variants (no SKU/GTIN/price/availability).
    Only used when JSON sources find nothing.
    """
    sizes: list[str] = []
    colors: list[str] = []
    for sig in dom_signals:
        if sig["type"] == "size" and not sizes:
            sizes = sig["values"]
        elif sig["type"] == "color" and not colors:
            colors = sig["values"]

    variants: list[Variant] = []
    if sizes and colors:
        for size in sizes:
            for color in colors:
                variants.append(Variant(attributes={"size": size, "color": color}))
    elif sizes:
        for size in sizes:
            variants.append(Variant(attributes={"size": size}))
    elif colors:
        for color in colors:
            variants.append(Variant(attributes={"color": color}))

    return variants


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


def _extract_fields_from_object(obj: dict, fields: dict, data_root: Any = None) -> None:
    """Extract Product fields from a product-like embedded JSON object."""
    _extract_name(obj, fields)
    _extract_brand(obj, fields)
    _extract_price_field(obj, fields)
    _extract_description(obj, fields)
    _extract_features(obj, fields)
    _extract_images_from_obj(obj, fields)
    _extract_video_from_obj(obj, fields)
    _extract_colors(obj, fields)
    _extract_variants(obj, fields, data_root=data_root)
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

    # Recursive collection from nested data (skip related/recommended product sections)
    recursive_urls = _collect_image_urls_recursive(obj, skip_related_keys=True)
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


def _extract_variants(obj: dict, fields: dict, data_root: Any = None) -> None:
    if fields.get("variants"):
        return

    # Generic: questions + skus cross-reference (common quiz/dimension pattern)
    questions = obj.get("questions")
    skus_list = obj.get("skus")
    if isinstance(questions, list) and isinstance(skus_list, list) and questions and skus_list:
        prices_list = obj.get("prices")
        media_list = obj.get("media") if isinstance(obj.get("media"), list) else None
        variants = _build_variants_from_questions(questions, skus_list, prices_list, media_list)
        if variants:
            fields["variants"] = variants
            # Build variant dimensions from questions
            dims = _build_dimensions_from_questions(questions)
            if dims:
                fields["variant_dimensions"] = dims
            return

    # Generic: items with name/sku/ean (e.g., headless CMS product objects)
    items = obj.get("items")
    if isinstance(items, list) and items:
        # Check for relatedProducts (color variants with their own items/sizes)
        related = obj.get("relatedProducts")
        product_sku = obj.get("productSku")
        if isinstance(related, list) and related and product_sku:
            # Derive base URL from seo.url (may be on a parent in the data tree)
            base_url = None
            seo = obj.get("seo")
            if isinstance(seo, dict) and isinstance(seo.get("url"), str):
                base_url = seo["url"]
            if not base_url and data_root is not None:
                seo_url = _find_key_recursive(data_root, "seo")
                if isinstance(seo_url, dict) and isinstance(seo_url.get("url"), str):
                    base_url = seo_url["url"]
            all_variants = _build_variants_from_related_products(obj, related, product_sku, base_url)
            if all_variants:
                fields["variants"] = all_variants[0]
                if all_variants[1]:
                    fields["variant_dimensions"] = all_variants[1]
                return

        # Single-color items extraction
        variants: list[Variant] = []
        color_name = obj.get("variantName")
        for item in items:
            if not isinstance(item, dict):
                continue
            if not item.get("sku") and not item.get("ean"):
                continue
            attrs: dict[str, str] = {}
            if color_name and isinstance(color_name, str):
                attrs["color"] = color_name
            name = item.get("name")
            if name:
                attrs["size"] = str(name)
            sku = item.get("sku") or item.get("item")
            ean = item.get("ean") or item.get("upc") or item.get("barcode")
            img = _extract_variant_image(
                item.get("image") or item.get("imageUrl") or item.get("featured_image")
            )
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
                    image_url=img,
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

    # Derive variant_dimensions from extracted variants if not already set
    if fields.get("variants") and not fields.get("variant_dimensions"):
        fields["variant_dimensions"] = _build_dimensions_from_variants(fields["variants"])


def _build_variants_from_generic_array(arr: list[dict]) -> list[Variant]:
    """Build Variant objects from a generic array of variant-like dicts."""
    variants: list[Variant] = []
    for item in arr:
        if not isinstance(item, dict):
            continue

        attrs: dict[str, str] = {}
        sku = None
        gtin = None
        image_url = None
        v_url = None
        available = True
        v_price = None
        v_image_urls: list[str] = []

        # Map of (regex, canonical_name) for all extractable attribute dimensions
        dim_patterns = [
            (_VARIANT_SIZE_KEYS, "size"),
            (_VARIANT_COLOR_KEYS, "color"),
            (_VARIANT_FIT_KEYS, "fit"),
            (_VARIANT_WIDTH_KEYS, "width"),
            (_VARIANT_LENGTH_KEYS, "length"),
            (_VARIANT_MATERIAL_KEYS, "material"),
            (_VARIANT_STYLE_KEYS, "style"),
            (_VARIANT_PATTERN_KEYS, "pattern"),
            (_VARIANT_FLAVOR_KEYS, "flavor"),
            (_VARIANT_STORAGE_KEYS, "storage"),
            (_VARIANT_FINISH_KEYS, "finish"),
        ]

        for k, v in item.items():
            # Check all dimension patterns
            dim_matched = False
            for pattern, dim_name in dim_patterns:
                if pattern.match(k):
                    if isinstance(v, str):
                        attrs[dim_name] = v
                    elif isinstance(v, dict) and v.get("name"):
                        attrs[dim_name] = v["name"]
                    elif isinstance(v, (int, float)):
                        attrs[dim_name] = str(v)
                    dim_matched = True
                    break

            if dim_matched:
                continue
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
            elif _VARIANT_IMAGE_KEYS.match(k) and image_url is None:
                image_url = _extract_variant_image(v)
            elif v_url is None and _VARIANT_URL_KEYS.match(k) and isinstance(v, str) and v.startswith(("http://", "https://", "/")):
                v_url = v

        # Shopify option1/option2/option3 pattern — positional variant attributes
        for opt_key in ("option1", "option2", "option3"):
            val = item.get(opt_key)
            if val and isinstance(val, str) and val.strip():
                # Map to a named dimension if we can infer from the options array
                options = item.get("options", [])
                opt_idx = int(opt_key[-1]) - 1
                if isinstance(options, list) and opt_idx < len(options):
                    dim_name = str(options[opt_idx]).lower().strip()
                else:
                    dim_name = opt_key  # fallback: "option1", "option2"
                if dim_name not in attrs:
                    attrs[dim_name] = val.strip()

        # Per-variant price
        if v_price is None:
            v_price = _extract_price_recursive(item, depth=5)

        # Per-variant images (merge single image extract with recursive collection)
        if image_url:
            v_image_urls = [image_url] + v_image_urls
        item_images = _collect_image_urls_recursive(item, max_depth=3)
        if item_images:
            v_image_urls = _normalize_and_dedup_urls(v_image_urls + item_images)

        if not attrs:
            name = item.get("name") or item.get("title")
            if name:
                attrs["size"] = str(name)

        if attrs or sku:
            variants.append(
                Variant(
                    attributes=attrs,
                    sku=sku,
                    gtin=gtin,
                    price=v_price,
                    image_urls=v_image_urls,
                    url=v_url,
                    available=available,
                )
            )

    return variants


def _build_variants_from_questions(
    questions: list, skus_list: list, prices_list: list | None = None, media_list: list | None = None
) -> list[Variant]:
    """Build variants from questions/answers + SKU cross-reference.

    Optionally follows price and media reference chains when available.
    """
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

    # Build price lookup from prices array (id -> Price)
    price_lookup: dict[str, Price] = {}
    if prices_list and isinstance(prices_list, list):
        for p in prices_list:
            if not isinstance(p, dict):
                continue
            pid = p.get("id") or p.get("priceId")
            amount = p.get("amount") or p.get("price")
            if pid and amount is not None:
                with contextlib.suppress(ValueError, TypeError):
                    price_lookup[str(pid)] = Price(
                        price=float(amount),
                        currency=p.get("currency", "USD"),
                        compare_at_price=float(p["compareAt"]) if p.get("compareAt") else None,
                    )

    # Build media lookup from media array (id -> image URL)
    media_lookup: dict[str, str] = {}
    if media_list and isinstance(media_list, list):
        for m in media_list:
            if not isinstance(m, dict):
                continue
            mid = m.get("id") or m.get("mediaId")
            src = m.get("src") or m.get("url")
            if mid and src and isinstance(src, str):
                media_lookup[str(mid)] = src

    variants: list[Variant] = []
    for sku in skus_list:
        if not isinstance(sku, dict):
            continue
        sku_id = str(sku.get("id", ""))
        attrs = sku_attrs.get(sku_id, {})
        if not attrs:
            continue

        image_url = _extract_variant_image(
            sku.get("image") or sku.get("imageUrl")
        )

        availability = sku.get("availability", {})
        available = True
        if isinstance(availability, dict):
            status = str(availability.get("status", "")).upper()
            available = status not in ("OUT_OF_STOCK", "UNAVAILABLE")

        # Per-variant price from reference chain
        v_price = None
        price_ref = sku.get("price")
        if isinstance(price_ref, str) and price_ref in price_lookup:
            v_price = price_lookup[price_ref]

        # Per-variant images from media references
        v_image_urls: list[str] = []
        media_refs = sku.get("media", [])
        if isinstance(media_refs, list):
            for ref in media_refs:
                ref_str = str(ref)
                if ref_str in media_lookup:
                    v_image_urls.append(media_lookup[ref_str])

        # Merge single image_url into image_urls list
        all_image_urls = v_image_urls
        if image_url and image_url not in all_image_urls:
            all_image_urls = [image_url] + all_image_urls

        variants.append(
            Variant(
                attributes=attrs,
                sku=sku_id,
                price=v_price,
                image_urls=all_image_urls,
                available=available,
            )
        )

    return variants


def _build_variants_from_related_products(
    main_obj: dict, related: list, product_sku: str, base_url: str | None = None
) -> tuple[list[Variant], list[VariantDimension]] | None:
    """Build color x size variant matrix from main product + relatedProducts.

    Some headless e-commerce platforms store sibling color variants as relatedProducts
    that share the same productSku. Each color variant has its own items (sizes),
    prices, and media.
    """
    # Collect all color product objects: main + related siblings
    color_products = [main_obj]
    for rp in related:
        if not isinstance(rp, dict):
            continue
        # Only include siblings with same productSku (not cross-sells)
        if str(rp.get("productSku", "")) == str(product_sku):
            color_products.append(rp)

    if len(color_products) < 2:
        return None

    # Derive URL prefix from base_url (strip last slug to get locale base)
    # e.g. "https://example.com/us/product-slug" -> "https://example.com/us/"
    url_prefix = ""
    if base_url:
        last_slash = base_url.rfind("/")
        if last_slash > len("https://x"):
            url_prefix = base_url[: last_slash + 1]

    variants: list[Variant] = []
    all_colors: list[str] = []
    all_sizes: list[str] = []

    for cp in color_products:
        color_name = cp.get("variantName", "")
        if not color_name or not isinstance(color_name, str):
            continue
        if color_name not in all_colors:
            all_colors.append(color_name)

        # Per-color price
        color_price = None
        cp_price = cp.get("price")
        if isinstance(cp_price, str):
            # Parse "170 USD" format
            parts = cp_price.strip().split()
            if len(parts) >= 1:
                with contextlib.suppress(ValueError):
                    amount = float(parts[0])
                    currency = parts[1] if len(parts) > 1 else "USD"
                    color_price = Price(price=amount, currency=currency)
        elif isinstance(cp_price, (int, float)):
            color_price = Price(price=float(cp_price), currency="USD")

        # Per-color images
        color_image_urls: list[str] = []
        media = cp.get("media")
        if isinstance(media, dict):
            # Try standard, full, or any resolution key
            for res_key in ("standard", "full", "max"):
                imgs = media.get(res_key)
                if isinstance(imgs, list):
                    color_image_urls.extend(str(u) for u in imgs if isinstance(u, str) and u.startswith("http"))
                    break
        media_objects = cp.get("mediaObjects")
        if not color_image_urls and isinstance(media_objects, list):
            for mo in media_objects:
                if isinstance(mo, dict):
                    sources = mo.get("sources")
                    if isinstance(sources, dict):
                        for src in sources.values():
                            if isinstance(src, str) and src.startswith("http"):
                                color_image_urls.append(src)
                                break

        # Per-color URL from URI slug
        color_url = None
        cp_uri = cp.get("uri")
        if url_prefix and isinstance(cp_uri, str) and cp_uri:
            color_url = url_prefix + cp_uri

        # Per-size items within this color
        items = cp.get("items", [])
        if not isinstance(items, list) or not items:
            continue

        cp_available = cp.get("available", True)
        for item in items:
            if not isinstance(item, dict):
                continue
            size_name = item.get("name")
            if not size_name:
                continue
            size_str = str(size_name)
            if size_str not in all_sizes:
                all_sizes.append(size_str)

            sku = item.get("sku") or item.get("item")
            ean = item.get("ean") or item.get("upc") or item.get("barcode")
            stock = item.get("stock")
            available = bool(cp_available)
            if isinstance(stock, (int, float)):
                available = stock > 0
            elif isinstance(stock, dict):
                total = sum(v for v in stock.values() if isinstance(v, (int, float)))
                available = total > 0

            variants.append(
                Variant(
                    attributes={"color": color_name, "size": size_str},
                    sku=str(sku) if sku else None,
                    gtin=str(ean) if ean else None,
                    price=color_price,
                    image_urls=color_image_urls,
                    url=color_url,
                    available=available,
                )
            )

    if not variants:
        return None

    dims: list[VariantDimension] = []
    if all_colors:
        dims.append(VariantDimension(name="color", values=all_colors))
    if all_sizes:
        dims.append(VariantDimension(name="size", values=all_sizes))

    return variants, dims


def _build_dimensions_from_questions(questions: list) -> list[VariantDimension]:
    """Build VariantDimension list from the questions/answers pattern."""
    from models import _ATTR_KEY_ALIASES

    dims: list[VariantDimension] = []
    for q in questions:
        if not isinstance(q, dict):
            continue
        q_type = str(q.get("type", "")).lower()
        if not q_type:
            continue
        # Apply the same normalization as Variant.attributes
        q_type = _ATTR_KEY_ALIASES.get(q_type, q_type)
        answers = q.get("answers", [])
        if not isinstance(answers, list):
            continue
        values = []
        for a in answers:
            if isinstance(a, dict) and a.get("title"):
                values.append(str(a["title"]))
        if values:
            dims.append(VariantDimension(name=q_type, values=values))
    return dims


def _build_dimensions_from_variants(variants: list[Variant]) -> list[VariantDimension]:
    """Derive VariantDimension list from existing variants' attributes."""
    dim_values: dict[str, list[str]] = {}
    for v in variants:
        for key, val in v.attributes.items():
            if key not in dim_values:
                dim_values[key] = []
            if val not in dim_values[key]:
                dim_values[key].append(val)
    return [VariantDimension(name=k, values=v) for k, v in dim_values.items()]


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
                # Schema.org standard properties + common extensions
                ld_dim_keys = {
                    "size": "size",
                    "color": "color",
                    "material": "material",
                    "pattern": "pattern",
                    "width": "width",
                    "depth": "depth",
                    "height": "height",
                    "weight": "weight",
                    "capacity": "storage",
                    "model": "style",
                }
                for ld_key, dim_name in ld_dim_keys.items():
                    val = v.get(ld_key)
                    if val:
                        attrs[dim_name] = str(val)
                if not attrs:
                    continue
                v_price = None
                v_url = None
                v_available = True
                v_offers = v.get("offers")
                if isinstance(v_offers, list) and v_offers:
                    v_offers = v_offers[0]
                if isinstance(v_offers, dict):
                    if "price" in v_offers:
                        v_price = Price(
                            price=float(v_offers["price"]),
                            currency=v_offers.get("priceCurrency", "USD"),
                        )
                    if v_offers.get("url"):
                        v_url = str(v_offers["url"])
                    avail = v_offers.get("availability", "")
                    if isinstance(avail, str) and "OutOfStock" in avail:
                        v_available = False
                # Fallback URL from @id (often has per-variant fragment like #size-5)
                if not v_url and v.get("@id") and isinstance(v.get("@id"), str):
                    v_url = v["@id"]
                # Per-variant image
                v_image_urls: list[str] = []
                v_image = v.get("image")
                if isinstance(v_image, str) and v_image.startswith("http"):
                    v_image_urls = [v_image]
                elif isinstance(v_image, list):
                    v_image_urls = [str(i) for i in v_image if isinstance(i, str) and i.startswith("http")]
                # GTIN: check gtin, gtin13, gtin14, gtin12
                v_gtin = v.get("gtin") or v.get("gtin13") or v.get("gtin14") or v.get("gtin12")
                variants.append(
                    Variant(
                        attributes=attrs,
                        sku=v.get("mpn") or v.get("sku"),
                        gtin=str(v_gtin) if v_gtin else None,
                        price=v_price,
                        image_urls=v_image_urls,
                        url=v_url,
                        available=v_available,
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

    # Ensure variant_dimensions are dicts
    if "variant_dimensions" in fields:
        fields["variant_dimensions"] = [
            d.model_dump() if isinstance(d, VariantDimension) else d for d in fields["variant_dimensions"]
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

    Excludes brand (brand names cause false matches in taxonomy).
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
    fields.setdefault("variant_dimensions", [])
    fields.setdefault("video_url", None)
    fields.setdefault("image_urls", [])
    fields.setdefault("description", "")


# =====================================================================
# Step 8b: Media ID Resolution for Variant Images
# =====================================================================

# Keys on SKU/variant objects that reference media IDs
_MEDIA_REF_KEYS = re.compile(r"(?i)^(mediaIds|media_ids|mediaId|imageIds|image_ids)$")


def _resolve_variant_media_ids(
    variants: list[Variant],
    embedded_json: dict[str, Any],
    media_lookups: list[list[dict]],
) -> list[Variant]:
    """Resolve media IDs on variants to actual image URLs.

    When SKU objects have a mediaIds field referencing a media lookup table,
    resolve the first matching ID to its src URL and assign to the variant.
    """
    if not media_lookups:
        return variants

    # Build unified lookup: media_id -> src_url
    lookup: dict[str, str] = {}
    for table in media_lookups:
        for entry in table:
            mid = entry.get("id", "")
            src = entry.get("src", "")
            if mid and src:
                lookup[mid] = src

    if not lookup:
        return variants

    # Build sku_id -> first_resolved_image_url
    sku_image_map = _build_sku_media_map(embedded_json, lookup)
    if not sku_image_map:
        return variants

    updated: list[Variant] = []
    for v in variants:
        if v.image_url is not None or not v.sku:
            updated.append(v)
            continue
        img = sku_image_map.get(v.sku)
        if img:
            updated.append(v.model_copy(update={"image_url": img}))
        else:
            updated.append(v)

    return updated


def _build_sku_media_map(
    embedded_json: dict[str, Any],
    media_lookup: dict[str, str],
) -> dict[str, str]:
    """Build a mapping from SKU ID to resolved image URL.

    Walks embedded JSON for SKU-like arrays where items have both an
    ID field and a media-reference field (list of strings matching
    media lookup IDs).
    """
    result: dict[str, str] = {}

    def walk(data: Any, depth: int = 0) -> None:
        if depth > 8:
            return
        if isinstance(data, list):
            if len(data) >= 2 and isinstance(data[0], dict):
                _try_resolve_sku_array(data, media_lookup, result)
            for item in data:
                walk(item, depth + 1)
        elif isinstance(data, dict):
            for v in data.values():
                walk(v, depth + 1)

    for _key, data in embedded_json.items():
        walk(data)

    return result


def _try_resolve_sku_array(
    arr: list[dict],
    media_lookup: dict[str, str],
    result: dict[str, str],
) -> None:
    """Check if array items have ID + media-reference fields, resolve them."""
    sample = arr[0]

    # Find the ID field
    id_key = None
    for k in sample:
        if _VARIANT_SKU_KEYS.match(k):
            id_key = k
            break
    if not id_key:
        return

    # Find the media reference field (list of strings matching lookup IDs)
    media_ref_key = None
    for k in sample:
        if _MEDIA_REF_KEYS.match(k):
            val = sample[k]
            if (
                isinstance(val, list)
                and val
                and isinstance(val[0], str)
                and any(mid in media_lookup for mid in val[:5])
            ):
                media_ref_key = k
                break
    if not media_ref_key:
        return

    for item in arr:
        sku_id = str(item.get(id_key, ""))
        refs = item.get(media_ref_key, [])
        if not sku_id or not isinstance(refs, list):
            continue
        for mid in refs:
            if isinstance(mid, str) and mid in media_lookup:
                src = media_lookup[mid]
                # Relaxed check: media lookup URLs are from a trusted source
                # (the page's own media table), so we don't require a file
                # extension — Akamai/CDN image servers often omit them.
                if _looks_like_media_src(src):
                    result[sku_id] = ("https:" + src) if src.startswith("//") else src
                    break


# =====================================================================
# Step 8c: Active Selection Injection into Variants
# =====================================================================


def _inject_active_selections(
    variants: list[Variant],
    active_selections: list[dict],
) -> list[Variant]:
    """Inject active dimension values into variants that are missing them.

    When a page represents a specific value of some dimension (e.g., color="Iron"),
    but extracted variants only have other dimensions (e.g., size), inject the
    active dimension into each variant's attributes.
    """
    if not variants or not active_selections:
        return variants

    updated = list(variants)
    for selection in active_selections:
        dim = selection.get("dimension", "")
        value = selection.get("value", "")
        if not dim or not value:
            continue

        # Skip if this dimension already exists on >50% of variants
        has_dim_count = sum(
            1 for v in updated
            if dim in {k.lower() for k in v.attributes}
        )
        if has_dim_count > len(updated) * 0.5:
            continue

        new_updated = []
        for v in updated:
            if dim not in {k.lower() for k in v.attributes}:
                new_attrs = dict(v.attributes)
                new_attrs[dim] = value
                new_updated.append(v.model_copy(update={"attributes": new_attrs}))
            else:
                new_updated.append(v)
        updated = new_updated

    return updated


# =====================================================================
# Step 8d: Sibling Variant Extraction
# =====================================================================

# Label fields on sibling variant objects
_SIBLING_LABEL_KEYS = re.compile(
    r"(?i)^(colorDescription|color_description|variantName|variant_name|"
    r"title|name|label|displayName)$"
)
# URL fields
_SIBLING_URL_KEYS = re.compile(
    r"(?i)^(pdpUrl|url|href|path|uri|link|productUrl|product_url)$"
)
# Image fields
_SIBLING_IMAGE_KEYS = re.compile(
    r"(?i)^(squarishImg|portraitImg|image|imageUrl|image_url|thumbnail|"
    r"thumbnailUrl|swatch|swatchImage|img)$"
)
# SKU/ID fields
_SIBLING_SKU_KEYS_RE = re.compile(
    r"(?i)^(styleColor|style_color|sku|productId|product_id|variantId|variant_id)$"
)
# Swatch object fields (nested dimension value)
_SIBLING_SWATCH_KEYS = re.compile(
    r"(?i)^(color_swatch|colour_swatch|swatch|colorSwatch|finish_swatch|material_swatch)$"
)
# Array key names that hint at sibling variants
_SIBLING_ARRAY_KEYS = re.compile(
    r"(?i)(colorway|variant.?product|related.?variant|sibling|"
    r"alternate.?color|other.?color|color.?option|colour.?option)"
)
# Dimension hint patterns for field names
_SIBLING_DIMENSION_HINTS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"(?i)color|colour"), "color"),
    (re.compile(r"(?i)finish"), "finish"),
    (re.compile(r"(?i)material"), "material"),
    (re.compile(r"(?i)pattern"), "pattern"),
    (re.compile(r"(?i)style"), "style"),
]


def _extract_sibling_variants(
    embedded_json: dict[str, Any],
    active_selections: list[dict],
    page_url: str,
) -> list[Variant]:
    """Extract sibling variants from embedded JSON arrays.

    Sibling variants represent the same product in different dimension values
    (e.g., different colorways). Each sibling has a label and optionally
    a URL, image, and SKU.
    """
    candidates = _find_sibling_arrays(embedded_json)
    if not candidates:
        return []

    # Pick the best candidate (most entries with richest fields)
    best = max(candidates, key=lambda arr: len(arr) * _sibling_richness(arr[0]))

    # Siblings must have varied labels — if all labels are the same,
    # this is a media gallery or similar repeated structure, not variants
    unique_labels = {_extract_sibling_label(item) for item in best[:20]} - {None}
    if len(unique_labels) < 2:
        return []

    dimension = _infer_sibling_dimension(best)

    # Build set of current-page values to exclude
    current_values: set[str] = set()
    for sel in active_selections:
        current_values.add(sel["value"].lower())
    if page_url:
        current_values.add(page_url.lower())

    variants: list[Variant] = []
    for item in best:
        label = _extract_sibling_label(item)
        if not label:
            continue

        # Skip if this is the current page's variant
        if _is_current_sibling(item, label, current_values, page_url):
            continue

        image = _extract_sibling_image(item)
        sku = _extract_sibling_sku(item)

        variants.append(Variant(
            attributes={dimension: label},
            sku=sku,
            image_url=image,
        ))

    return variants


def _find_sibling_arrays(
    data: Any,
    depth: int = 0,
) -> list[list[dict]]:
    """Find arrays of dicts that look like sibling variant lists."""
    results: list[list[dict]] = []
    if depth > 8:
        return results
    if isinstance(data, dict):
        for k, v in data.items():
            if not isinstance(v, list) or len(v) < 2:
                _find_sibling_arrays_recurse(v, results, depth)
                continue
            if not all(isinstance(item, dict) for item in v[:3]):
                _find_sibling_arrays_recurse(v, results, depth)
                continue
            # Check by key name hint
            if isinstance(k, str) and _SIBLING_ARRAY_KEYS.search(k) and _has_sibling_fields(v[0]):
                results.append(v)
                continue
            # Check by field content
            if _looks_like_sibling_array(v):
                results.append(v)
                continue
            _find_sibling_arrays_recurse(v, results, depth)
    elif isinstance(data, list):
        for item in data:
            results.extend(_find_sibling_arrays(item, depth + 1))
    return results


def _find_sibling_arrays_recurse(
    data: Any,
    results: list[list[dict]],
    depth: int,
) -> None:
    """Helper to recurse into non-sibling values."""
    if isinstance(data, (dict, list)):
        results.extend(_find_sibling_arrays(data, depth + 1))


def _looks_like_sibling_array(arr: list) -> bool:
    """Check if array items look like sibling variant objects."""
    if not arr or not isinstance(arr[0], dict):
        return False
    # Must have sibling-like fields but NOT be a regular variant array
    if not _has_sibling_fields(arr[0]):
        return False
    return not _looks_like_variant_array(arr)


def _has_sibling_fields(d: dict) -> bool:
    """Check if a dict has the field pattern of a sibling variant.

    Requires a label AND at least one product-specific indicator (image, SKU,
    or swatch). A label + URL alone is too generic — navigation link arrays
    also have {name, url} and would cause false positives.
    """
    keys = set(d.keys())
    has_label = any(_SIBLING_LABEL_KEYS.match(k) for k in keys)
    has_image = any(_SIBLING_IMAGE_KEYS.match(k) for k in keys)
    has_sku = any(_SIBLING_SKU_KEYS_RE.match(k) for k in keys)
    has_swatch = any(_SIBLING_SWATCH_KEYS.match(k) for k in keys)
    return has_label and (has_image or has_sku or has_swatch)


def _sibling_richness(d: dict) -> int:
    """Score how rich a sibling dict is (more useful fields = better)."""
    score = 0
    for k in d:
        if _SIBLING_LABEL_KEYS.match(k) or _SIBLING_IMAGE_KEYS.match(k):
            score += 2
        elif _SIBLING_SKU_KEYS_RE.match(k) or _SIBLING_SWATCH_KEYS.match(k):
            score += 1
    return score


def _infer_sibling_dimension(arr: list[dict]) -> str:
    """Infer the dimension name from sibling array field names."""
    if not arr:
        return "color"
    sample = arr[0]
    # Check field names for dimension hints
    for k in sample:
        for pattern, dim in _SIBLING_DIMENSION_HINTS:
            if pattern.search(k):
                return dim
    # Check swatch objects
    for k, _v in sample.items():
        if _SIBLING_SWATCH_KEYS.match(k):
            for pattern, dim in _SIBLING_DIMENSION_HINTS:
                if pattern.search(k):
                    return dim
    return "color"


def _extract_sibling_label(item: dict) -> str | None:
    """Extract the human-readable label from a sibling dict."""
    # Priority 1: swatch object with name
    for k, v in item.items():
        if _SIBLING_SWATCH_KEYS.match(k) and isinstance(v, dict):
            name = v.get("name") or v.get("label") or v.get("title")
            if isinstance(name, str) and name.strip():
                return name.strip()
    # Priority 2: direct label fields
    for k, v in item.items():
        if _SIBLING_LABEL_KEYS.match(k) and isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _extract_sibling_image(item: dict) -> str | None:
    """Extract an image URL from a sibling dict."""
    for k, v in item.items():
        if _SIBLING_IMAGE_KEYS.match(k) and isinstance(v, str) and _looks_like_image_url_string(v):
            return ("https:" + v) if v.startswith("//") else v
    return None


def _extract_sibling_sku(item: dict) -> str | None:
    """Extract a SKU/ID from a sibling dict."""
    for k, v in item.items():
        if _SIBLING_SKU_KEYS_RE.match(k) and isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _is_current_sibling(
    item: dict,
    label: str,
    current_values: set[str],
    page_url: str,
) -> bool:
    """Check if a sibling entry represents the current page."""
    if label.lower() in current_values:
        return True
    # Check URL match
    for k, v in item.items():
        if _SIBLING_URL_KEYS.match(k) and isinstance(v, str):
            if page_url and v.lower().rstrip("/") == page_url.lower().rstrip("/"):
                return True
            # Also check if URL slug is contained in page_url
            if page_url and v.strip("/").lower() in page_url.lower():
                return True
    return False


# =====================================================================
# Stage D: Vision-based QA Validation
# =====================================================================

_QA_SYSTEM_PROMPT = (
    "Product data QA validator with vision. You see extracted data AND the actual product images.\n\n"
    "IMAGE INSPECTION (critical):\n"
    "- For each numbered image, decide KEEP or REMOVE.\n"
    "- KEEP: actual photo of THIS product (hero shots, detail shots, lifestyle with product visible, in-context shots).\n"
    "- REMOVE: banners, icons, logos, size charts, shipping/return graphics, review photos, "
    "UI elements, lifestyle shots where the product is NOT visible, cross-sell images of OTHER products, swatch thumbnails.\n"
    "- For each KEPT image, identify which color variant it shows using the product's color list. "
    "Set null if product has one color or color is ambiguous.\n\n"
    "FIELD CHECKS:\n"
    "- COLORS: Must be real color names. Flag hex codes, CSS values, SKU codes, 'Default', 'N/A', 'Choose a design'.\n"
    "- KEY_FEATURES: Must be product-specific. Flag duplicates, nav text, disclaimers.\n"
    "- VARIANTS: Flag if duplicate attribute combinations exist.\n"
    "- NAME/BRAND/CATEGORY/PRICE: Flag only clear errors.\n\n"
    "Return image_verdicts for EVERY image. Set passed=true only if no images removed and no field issues."
)


def _build_qa_content(product: Product, parsed: ParsedPage) -> list[dict]:
    """Build multimodal content: product summary text + actual images for vision inspection."""
    parts: list[str] = [
        f"Product: {product.name}",
        f"Brand: {product.brand}",
        f"Price: {product.price.price} {product.price.currency}",
    ]
    if product.price.compare_at_price:
        parts.append(f"Was: {product.price.compare_at_price}")
    parts.append(f"Category: {product.category.name}")
    parts.append(f"Colors: {product.colors}")
    if product.key_features:
        parts.append(f"Features: {'; '.join(product.key_features[:5])}")
    parts.append(f"Variants: {len(product.variants)}")
    for v in product.variants[:5]:
        parts.append(f"  {v.attributes}")
    if len(product.variants) > 5:
        parts.append(f"  ...+{len(product.variants) - 5} more")

    summary = "\n".join(parts)

    # Build multimodal content blocks: text summary + numbered images
    content: list[dict] = [
        {"type": "input_text", "text": f"QA this product:\n\n{summary}\n\nImages ({len(product.image_urls)}):"},
    ]
    for i, url in enumerate(product.image_urls):
        content.append({"type": "input_text", "text": f"[Image {i}]"})
        content.append({"type": "input_image", "image_url": url})

    return content


def _apply_qa_corrections(product: Product, result: QAResult) -> tuple[Product, int]:
    """Apply vision QA corrections. Returns (corrected product, num corrections applied)."""
    updates: dict[str, object] = {}
    applied = 0

    # --- Image filtering and color-image mapping ---
    if result.image_verdicts:
        keep_indices: set[int] = set()
        color_to_urls: dict[str, list[str]] = {}

        for v in result.image_verdicts:
            if 0 <= v.index < len(product.image_urls):
                if v.keep:
                    keep_indices.add(v.index)
                    if v.color:
                        color_to_urls.setdefault(v.color, []).append(product.image_urls[v.index])

        # Only filter if the model actually removed some images
        removed = len(product.image_urls) - len(keep_indices)
        if removed > 0 and keep_indices:
            updates["image_urls"] = [
                url for i, url in enumerate(product.image_urls) if i in keep_indices
            ]
            applied += 1
            logger.info(f"  QA removed {removed} non-product images, kept {len(keep_indices)}")

        # Assign images to variants that have a color but no image
        if color_to_urls and product.variants:
            updated_variants = []
            variants_updated = 0
            for var in product.variants:
                color_attr = var.attributes.get("color", "")
                if color_attr and not var.image_url:
                    # Find matching color in the map (case-insensitive)
                    for color_name, urls in color_to_urls.items():
                        if color_name.lower() == color_attr.lower() and urls:
                            var = var.model_copy(update={"image_url": urls[0]})
                            variants_updated += 1
                            break
                updated_variants.append(var)
            if variants_updated > 0:
                updates["variants"] = updated_variants
                applied += 1
                logger.info(f"  QA matched images to {variants_updated} variants")

    # --- Field corrections ---
    if result.name and result.name.strip() and result.name != product.name:
        updates["name"] = result.name.strip()
        applied += 1

    if result.brand and result.brand.strip() and result.brand != product.brand:
        updates["brand"] = result.brand.strip()
        applied += 1

    if result.category and result.category in VALID_CATEGORIES and result.category != product.category.name:
        updates["category"] = Category(name=result.category)
        applied += 1

    if result.colors is not None:
        clean = [c for c in result.colors if isinstance(c, str) and c.strip()]
        if clean:
            updates["colors"] = clean
            applied += 1

    if result.key_features is not None:
        clean = [f for f in result.key_features if isinstance(f, str) and f.strip()]
        if clean:
            updates["key_features"] = clean
            applied += 1

    if result.has_duplicate_variants and product.variants:
        seen: set[str] = set()
        deduped: list[Variant] = []
        for v in product.variants:
            key = str(sorted(v.attributes.items()))
            if key not in seen:
                seen.add(key)
                deduped.append(v)
        if len(deduped) < len(product.variants):
            updates["variants"] = updated_variants if "variants" in updates else deduped
            applied += 1

    if not updates:
        return product, 0

    return product.model_copy(update=updates), applied


async def _qa_check(product: Product, parsed: ParsedPage) -> tuple[Product, list[str]]:
    """Run vision-based LLM QA on a fully-formed Product.

    Sends actual product images to a multimodal model for inspection.
    Returns (possibly corrected product, list of issue descriptions).
    """
    if not QA_ENABLED:
        return product, []

    try:
        content = _build_qa_content(product, parsed)
        result = await ai.responses(
            model=LLM_MODEL,
            input=[
                {"role": "system", "content": _QA_SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            text_format=QAResult,
        )

        issues = result.issues or []
        # Count image removals as issues too
        removed_count = sum(1 for v in result.image_verdicts if not v.keep)
        if removed_count:
            issues.append(f"Removed {removed_count} non-product images")

        if result.passed and removed_count == 0:
            return product, []

        corrected, num_applied = _apply_qa_corrections(product, result)
        logger.info(f"  QA applied {num_applied} corrections")
        return corrected, issues

    except Exception as e:
        logger.warning(f"  QA check failed (skipping): {e}")
        return product, []
