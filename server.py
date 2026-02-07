"""
FastAPI server for the product catalog.

Loads products.json at startup into memory and serves two endpoints:
- GET /api/products     → slim product cards for catalog grid
- GET /api/products/{slug} → full product detail
"""

import asyncio
import logging
import re
from pathlib import Path

import httpx
import orjson
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel

from models import Price, Variant, VisualVariant

# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class ProductCard(BaseModel):
    """Slim payload for catalog grid cards."""

    slug: str
    name: str
    brand: str
    price: float
    currency: str
    compare_at_price: float | None
    thumbnail_url: str | None
    category: str
    color_count: int
    variant_count: int
    colors: list[str]


class ProductDetail(BaseModel):
    """Full payload for the product detail page."""

    slug: str
    name: str
    brand: str
    price: Price
    description: str
    key_features: list[str]
    image_urls: list[str]
    video_url: str | None
    category: str
    category_breadcrumbs: list[str]
    colors: list[str]
    variants: list[Variant]
    visual_variants: list[VisualVariant]


# ---------------------------------------------------------------------------
# Slug generation
# ---------------------------------------------------------------------------

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slugify(text: str) -> str:
    """Convert product name to URL-safe slug."""
    return _SLUG_RE.sub("-", text.lower()).strip("-")


# ---------------------------------------------------------------------------
# Image curation
# ---------------------------------------------------------------------------

# Leading numeric product-ID prefix to strip for secondary dedup.
# Catches patterns like "224626_1176_41" vs "224625_1176_41" where the
# leading segment is a product/SKU ID and the rest identifies color+view.
_LEADING_ID_PREFIX = re.compile(r"^\d+[_-](?=\d)")

# Common CDN size/transform suffixes to strip when deduplicating
_SIZE_SUFFIXES = re.compile(
    r"[-_](mini|thumb|thumbnail|square|small|medium|large|full|max|"
    r"\d+x\d+|\d+x0|0x\d+|2x3|1x1|crop|resize|scaled|optimized)"
    r"(?=\.[a-z]{3,4}(?:\?|$))",
    re.IGNORECASE,
)

# File extensions that are never product photos
_NON_PHOTO_EXTENSIONS = re.compile(r"\.(svg|gif|ico|bmp)(\?|$)", re.IGNORECASE)

# Generic non-product URL path patterns (site chrome, navigation, promotions)
_NON_PRODUCT_PATH = re.compile(
    r"(flyout|menu[-_]|banner|[-_/]icon[s]?[-_./]|[-_/]nav[-_./]|footer|header|sidebar|"
    r"spinner|loading|spacer|pixel|tracking|analytics|"
    r"badge|widget|placeholder|overlay|associate|"
    r"[-_/]ad[-_./]|promo|"
    r"product_review_image|review[-_]image|customer[-_]photo|"
    r"cross[-_]sell|recommendation)",
    re.IGNORECASE,
)


def _is_likely_product_image(url: str) -> bool:
    """Return False for URLs that are almost certainly not product photos."""
    path = url.split("?")[0]
    if _NON_PHOTO_EXTENSIONS.search(path):
        return False
    return not _NON_PRODUCT_PATH.search(path)


def _suffix_quality_score(filename: str) -> int:
    """Score a filename by its CDN size suffix — higher means larger/better.

    Used to pick the best variant when multiple URLs map to the same base.
    """
    match = _SIZE_SUFFIXES.search(filename)
    if not match:
        return 50  # No suffix — unknown/default size

    suffix = match.group(1).lower()

    scores = {
        "max": 100,
        "full": 90,
        "large": 80,
        "optimized": 60,
        "scaled": 55,
        "medium": 40,
        "square": 35,
        "crop": 35,
        "resize": 35,
        "small": 20,
        "thumb": 10,
        "thumbnail": 10,
        "mini": 5,
    }
    if suffix in scores:
        return scores[suffix]

    # Dimension descriptors (e.g. 200x0, 800x600, 0x400)
    try:
        parts = suffix.split("x")
        w = int(parts[0]) if parts[0] != "0" else int(parts[1])
        return w // 10  # 200→20, 800→80, 1200→120
    except (ValueError, IndexError):
        return 30


# ---------------------------------------------------------------------------
# CDN URL upgrades — request the highest-resolution variant available
# ---------------------------------------------------------------------------

# Akamai Image Server: ?wid=NNN query param
_AKAMAI_WID = re.compile(r"([?&])wid=\d+")


def _upgrade_cdn_url(url: str) -> str:
    """Upgrade a CDN image URL to request the highest available resolution."""
    # Akamai Image Server → request large width
    if "/is/image/" in url:
        if _AKAMAI_WID.search(url):
            return _AKAMAI_WID.sub(r"\g<1>wid=1200", url)
        sep = "&" if "?" in url else "?"
        return url + sep + "wid=1200"

    return url


def _curate_images(image_urls: list[str], max_images: int = 12) -> list[str]:
    """Filter non-product images, deduplicate by base filename, cap at max_images.

    When multiple URLs map to the same base filename (e.g. image-full.jpg and
    image-mini.jpg), the URL with the highest quality suffix wins.

    A secondary dedup pass strips leading numeric product-ID prefixes so that
    URLs like ``224626_1176_41`` and ``224625_1176_41`` (same image from
    different SKU variants) are treated as duplicates.
    """
    # base -> (url, quality_score)
    best_for_base: dict[str, tuple[str, int]] = {}
    # Track insertion order of first-seen bases
    base_order: list[str] = []
    # Secondary dedup: tail key (leading product-ID stripped) → first base
    tail_seen: dict[str, str] = {}

    for url in image_urls:
        if not _is_likely_product_image(url):
            continue

        path = url.split("?")[0]
        filename = path.rsplit("/", 1)[-1] if "/" in path else path
        base = _SIZE_SUFFIXES.sub("", filename).lower()
        score = _suffix_quality_score(filename)

        # Secondary key: strip leading product-ID prefix for cross-SKU dedup
        tail = _LEADING_ID_PREFIX.sub("", base)
        if tail != base and tail in tail_seen:
            # Already seen an image with the same tail — treat as duplicate
            existing_base = tail_seen[tail]
            if existing_base in best_for_base and score > best_for_base[existing_base][1]:
                best_for_base[existing_base] = (url, score)
            continue

        if base not in best_for_base:
            best_for_base[base] = (url, score)
            base_order.append(base)
            if tail != base:
                tail_seen[tail] = base
        elif score > best_for_base[base][1]:
            best_for_base[base] = (url, score)

    curated: list[str] = []
    for base in base_order:
        curated.append(_upgrade_cdn_url(best_for_base[base][0]))
        if len(curated) >= max_images:
            break

    return curated


# ---------------------------------------------------------------------------
# Visual variant construction
# ---------------------------------------------------------------------------


def _color_slugs(color: str) -> list[str]:
    """Generate URL slug variants for a color name.

    "Jet Black" → ["jet-black", "jet_black", "jetblack"]
    """
    lower = color.lower()
    return [
        _SLUG_RE.sub("-", lower).strip("-"),
        re.sub(r"[^a-z0-9]+", "_", lower).strip("_"),
        re.sub(r"[^a-z0-9]", "", lower),
    ]


def _build_visual_variants(product: dict, image_urls: list[str]) -> list[VisualVariant]:
    """Build visual variants by matching images to colors.

    Uses two strategies:
      1. Variant-level images: variants with a color attribute that have image_url
      2. URL filename matching: product-level images containing a color slug in the path

    Returns visual variants only if >= 2 colors have matching images.
    """
    colors = product.get("colors", [])
    if not colors or not image_urls:
        return []

    # Strategy 1: collect variant-level images grouped by color
    variant_images: dict[str, list[str]] = {}
    for v in product.get("variants", []):
        color = v.get("attributes", {}).get("color", "").strip()
        img = v.get("image_url")
        if color and img:
            variant_images.setdefault(color, []).append(img)

    # Strategy 2: match product-level images to colors by URL filename
    color_to_images: dict[str, list[str]] = {}
    for color in colors:
        slugs = _color_slugs(color)
        matched: list[str] = []

        # Start with variant-level images for this color
        if color in variant_images:
            matched.extend(variant_images[color])

        # Then add URL-matched product-level images
        for url in image_urls:
            path = url.split("?")[0].lower()
            if any(s and s in path for s in slugs) and url not in matched:
                matched.append(url)

        if matched:
            color_to_images[color] = matched

    # Need at least 2 colors with images for interactive switching
    if len(color_to_images) < 2:
        return []

    # Build visual variants with per-color curation (preserving original color order)
    visual_variants: list[VisualVariant] = []
    for color in colors:
        if color in color_to_images:
            curated = _curate_images(color_to_images[color])
            if curated:
                visual_variants.append(
                    VisualVariant(
                        label=color,
                        slug=_slugify(color),
                        image_urls=curated,
                    )
                )

    return visual_variants if len(visual_variants) >= 2 else []


# ---------------------------------------------------------------------------
# Image validation via HEAD requests
# ---------------------------------------------------------------------------

_PHOTO_CONTENT_TYPES = frozenset({"image/jpeg", "image/png", "image/webp", "image/avif"})
_MIN_IMAGE_BYTES = 15_360  # 15 KB — product photos are larger than icons/badges/small promos

logger = logging.getLogger("server")


async def _validate_image_urls(all_urls: set[str]) -> set[str]:
    """HEAD-check image URLs in parallel; return those that look like product photos."""
    approved: set[str] = set()
    semaphore = asyncio.Semaphore(20)

    async def _check(client: httpx.AsyncClient, url: str) -> None:
        try:
            async with semaphore:
                resp = await client.head(url, follow_redirects=True, timeout=5.0)
            if resp.status_code != 200:
                return
            content_type = resp.headers.get("content-type", "").split(";")[0].strip().lower()
            if content_type not in _PHOTO_CONTENT_TYPES:
                return
            content_length = resp.headers.get("content-length")
            if content_length and int(content_length) < _MIN_IMAGE_BYTES:
                return
            approved.add(url)
        except (httpx.HTTPError, ValueError):
            # Network error or bad header — skip this URL
            pass

    async with httpx.AsyncClient() as client:
        await asyncio.gather(*[_check(client, url) for url in all_urls])

    return approved


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

PRODUCTS_FILE = Path(__file__).parent / "products.json"

# In-memory stores populated at startup
_products_by_slug: dict[str, dict] = {}
_product_cards: list[ProductCard] = []


async def _load_products() -> None:
    """Load products.json into memory, validate image URLs, build lookups."""
    global _products_by_slug, _product_cards

    if not PRODUCTS_FILE.exists():
        raise FileNotFoundError(
            f"{PRODUCTS_FILE} not found. Run 'python main.py' first to extract products."
        )

    raw = PRODUCTS_FILE.read_bytes()
    products: list[dict] = orjson.loads(raw)

    # Collect all unique image URLs across all products for batch validation
    all_urls: set[str] = set()
    for product in products:
        all_urls.update(product.get("image_urls", []))

    # Validate via HEAD requests (graceful degradation on failure)
    try:
        approved = await _validate_image_urls(all_urls)
        logger.info("Image validation: %d/%d URLs approved", len(approved), len(all_urls))
    except Exception:
        logger.warning("Image validation failed, using unvalidated URLs", exc_info=True)
        approved = all_urls  # fall back to all URLs

    # Filter each product's images to only approved URLs
    for product in products:
        product["image_urls"] = [u for u in product.get("image_urls", []) if u in approved]

    by_slug: dict[str, dict] = {}
    cards: list[ProductCard] = []

    for product in products:
        slug = _slugify(product["name"])

        # Handle duplicate slugs by appending brand
        if slug in by_slug:
            slug = _slugify(f"{product['brand']}-{product['name']}")

        product["slug"] = slug
        by_slug[slug] = product

        # Build slim card (thumbnail uses first validated image)
        image_urls = product.get("image_urls", [])
        price_data = product.get("price", {})
        category_path = product.get("category", {}).get("name", "")
        leaf_category = category_path.rsplit(" > ", 1)[-1] if category_path else ""

        cards.append(
            ProductCard(
                slug=slug,
                name=product["name"],
                brand=product.get("brand", ""),
                price=price_data.get("price", 0),
                currency=price_data.get("currency", "USD"),
                compare_at_price=price_data.get("compare_at_price"),
                thumbnail_url=image_urls[0] if image_urls else None,
                category=leaf_category,
                color_count=len(product.get("colors", [])),
                variant_count=len(product.get("variants", [])),
                colors=product.get("colors", []),
            )
        )

    _products_by_slug = by_slug
    _product_cards = cards


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Product Catalog API",
    default_response_class=ORJSONResponse,
)

app.add_middleware(GZipMiddleware, minimum_size=500)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["GET"],
    allow_headers=["*"],
    max_age=86400,
)


@app.on_event("startup")
async def startup() -> None:
    await _load_products()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/api/products", response_model=list[ProductCard])
async def list_products():
    """Return slim product cards for the catalog grid."""
    return _product_cards


@app.get("/api/products/{slug}", response_model=ProductDetail)
async def get_product(slug: str):
    """Return full product detail for a single product."""
    product = _products_by_slug.get(slug)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    category_path = product.get("category", {}).get("name", "")
    breadcrumbs = [s.strip() for s in category_path.split(">")] if category_path else []

    all_image_urls = product.get("image_urls", [])
    visual_variants = _build_visual_variants(product, all_image_urls)

    # If visual variants exist, default gallery shows the first variant's images.
    # Otherwise fall back to global curation.
    if visual_variants:
        image_urls = visual_variants[0].image_urls
    else:
        image_urls = _curate_images(all_image_urls)

    return ProductDetail(
        slug=product["slug"],
        name=product["name"],
        brand=product.get("brand", ""),
        price=Price(**product["price"]),
        description=product.get("description", ""),
        key_features=product.get("key_features", []),
        image_urls=image_urls,
        video_url=product.get("video_url"),
        category=category_path,
        category_breadcrumbs=breadcrumbs,
        colors=product.get("colors", []),
        variants=[Variant(**v) for v in product.get("variants", [])],
        visual_variants=visual_variants,
    )
