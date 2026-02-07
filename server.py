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

from models import Price, Variant

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
    r"[-_/]ad[-_./]|promo)",
    re.IGNORECASE,
)


def _is_likely_product_image(url: str) -> bool:
    """Return False for URLs that are almost certainly not product photos."""
    path = url.split("?")[0]
    if _NON_PHOTO_EXTENSIONS.search(path):
        return False
    return not _NON_PRODUCT_PATH.search(path)


def _curate_images(image_urls: list[str], max_images: int = 12) -> list[str]:
    """Filter non-product images, deduplicate by base filename, cap at max_images."""
    seen_bases: set[str] = set()
    curated: list[str] = []

    for url in image_urls:
        if not _is_likely_product_image(url):
            continue

        # Extract filename from URL path (before query params)
        path = url.split("?")[0]
        filename = path.rsplit("/", 1)[-1] if "/" in path else path

        # Strip size suffixes for deduplication
        base = _SIZE_SUFFIXES.sub("", filename).lower()

        if base not in seen_bases:
            seen_bases.add(base)
            curated.append(url)
            if len(curated) >= max_images:
                break

    return curated


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

    return ProductDetail(
        slug=product["slug"],
        name=product["name"],
        brand=product.get("brand", ""),
        price=Price(**product["price"]),
        description=product.get("description", ""),
        key_features=product.get("key_features", []),
        image_urls=_curate_images(product.get("image_urls", [])),
        video_url=product.get("video_url"),
        category=category_path,
        category_breadcrumbs=breadcrumbs,
        colors=product.get("colors", []),
        variants=[Variant(**v) for v in product.get("variants", [])],
    )
