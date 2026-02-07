import re
from pathlib import Path

from pydantic import BaseModel, field_validator

# Load categories once at module level
CATEGORIES_FILE = Path(__file__).parent / "categories.txt"
VALID_CATEGORIES = set()
if CATEGORIES_FILE.exists():
    with open(CATEGORIES_FILE) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                # Parse "ID - Category > Path" format (strip numeric ID prefix)
                if " - " in line:
                    line = line.split(" - ", 1)[1]
                VALID_CATEGORIES.add(line)

# Canonical synonyms for variant attribute keys.
# Maps raw key (lowercased) -> canonical dimension name.
_ATTR_KEY_ALIASES = {
    # -> color
    "colour": "color",
    "colorway": "color",
    "shade": "color",
    "hue": "color",
    "color_name": "color",
    "colorname": "color",
    "selectedcolor": "color",
    "selected_color": "color",
    "swatch": "color",
    # -> size
    "sizing": "size",
    "shoe_size": "size",
    "shoesize": "size",
    "clothing_size": "size",
    "apparel_size": "size",
    "taille": "size",  # French
    # -> fit
    "item": "fit",  # some platforms use "item" for the fit dimension
    "cut": "fit",
    "silhouette": "fit",
    # -> width
    "shoe_width": "width",
    "shoewidth": "width",
    # -> length
    "inseam": "length",
    "leg_length": "length",
    # -> material
    "fabric": "material",
    "composition": "material",
    "textile": "material",
    # -> flavor
    "flavour": "flavor",
    "scent": "flavor",
    "fragrance": "flavor",
    # -> pattern
    "print": "pattern",
    "motif": "pattern",
    # -> style
    "edition": "style",
    "model": "style",
    "version": "style",
    # -> storage (electronics)
    "memory": "storage",
    "ram": "storage",
    # -> finish (furniture/hardware — distinct from color)
    "surface": "finish",
    "coating": "finish",
}


class Category(BaseModel):
    # A category from Google's Product Taxonomy
    # https://www.google.com/basepages/producttype/taxonomy.en-US.txt
    name: str

    @field_validator("name")
    @classmethod
    def validate_name_exists(cls, v: str) -> str:
        if v not in VALID_CATEGORIES:
            raise ValueError(f"Category '{v}' is not a valid category in categories.txt")
        return v


class Price(BaseModel):
    price: float
    currency: str
    # If a product is on sale, this is the original price
    compare_at_price: float | None = None


class VariantDimension(BaseModel):
    """A dimension along which a product varies (e.g., size, color)."""

    name: str  # normalized lowercase: "color", "size", "fit"
    values: list[str]  # ordered list of all possible values


class Variant(BaseModel):
    """A specific purchasable configuration of a product."""

    attributes: dict[str, str]  # e.g. {"size": "10", "color": "Black", "fit": "Regular"}
    sku: str | None = None
    gtin: str | None = None
    price: Price | None = None  # only if different from base product price
    image_urls: list[str] = []  # variant-specific images
    url: str | None = None  # variant-specific product page URL
    available: bool = True

    @field_validator("attributes")
    @classmethod
    def normalize_attribute_keys(cls, v: dict[str, str]) -> dict[str, str]:
        normalized = {}
        for key, val in v.items():
            key = key.lower().strip()
            key = _ATTR_KEY_ALIASES.get(key, key)
            normalized[key] = val
        return normalized

    @field_validator("gtin")
    @classmethod
    def validate_gtin_format(cls, v: str | None) -> str | None:
        if v is not None:
            digits = re.sub(r"\D", "", v)
            if digits and len(digits) not in (8, 12, 13, 14):
                return None  # drop malformed GTINs silently
        return v


class VisualVariant(BaseModel):
    """A visually distinct version of a product (e.g., a specific color).

    Built at serve time by matching product images to color names via URL patterns.
    Each visual variant gets its own curated image gallery.
    """

    label: str  # Human-readable: "Iron", "Navy Blue"
    slug: str  # URL-safe: "iron", "navy-blue"
    image_urls: list[str]  # Curated images for this color


# This is the final product schema that you need to output.
# You may add additional models as needed.
class Product(BaseModel):
    name: str
    price: Price
    description: str
    key_features: list[str]
    image_urls: list[str]
    video_url: str | None = None
    category: Category
    brand: str
    colors: list[str]
    variant_dimensions: list[VariantDimension] = []
    variants: list[Variant]
