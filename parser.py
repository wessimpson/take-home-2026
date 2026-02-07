"""
Generic HTML parser for product pages.

Extracts structured data from universal web standards:
JSON-LD, Open Graph meta tags, embedded JSON state objects,
visible body text, image URLs, and video URLs.

No site-specific or page-specific logic.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


@dataclass
class ParsedPage:
    """All structured data extracted from an HTML page."""

    json_ld: list[dict] = field(default_factory=list)
    og_tags: dict[str, str] = field(default_factory=dict)
    embedded_json: dict[str, Any] = field(default_factory=dict)
    body_text: str = ""
    image_urls: list[str] = field(default_factory=list)
    video_urls: list[str] = field(default_factory=list)
    meta_tags: dict[str, str] = field(default_factory=dict)
    breadcrumbs: list[str] = field(default_factory=list)  # ordered breadcrumb trail
    # Platform detection
    platform: str = "unknown"  # shopify, magento, sfcc, custom
    js_framework: str = "unknown"  # next.js, angular, remix, svelte, gatsby, etc.
    # DOM-based fallback signals
    dom_variants: list[dict] = field(default_factory=list)
    dom_sale_price: dict[str, Any] = field(default_factory=dict)
    # Active selections: which dimension values are currently selected on this page
    # Each dict: {"dimension": str, "value": str, "source": str}
    page_url: str = ""
    active_selections: list[dict] = field(default_factory=list)
    # Media lookup tables: arrays of {id, src} objects from embedded JSON
    media_lookups: list[list[dict]] = field(default_factory=list)


def parse_html(html: str) -> ParsedPage:
    """Parse an HTML page and extract all structured data sources."""
    soup = BeautifulSoup(html, "lxml")

    platform, js_framework = _detect_platform(html)
    json_ld = _extract_json_ld(soup)
    og_tags = _extract_og_tags(soup)
    embedded_json = _extract_embedded_json(soup, html)
    body_text = _extract_body_text(soup)
    meta_tags = _extract_meta_tags(soup)
    image_urls = _extract_image_urls(soup)
    video_urls = _extract_video_urls(html)
    breadcrumbs = _extract_breadcrumbs(json_ld, soup)
    dom_variants = _extract_dom_variants(soup)
    dom_sale_price = _extract_dom_sale_price(soup)
    page_url = _extract_page_url(soup, og_tags)
    active_selections = _extract_active_selections(soup, embedded_json)
    media_lookups = _extract_media_lookups(embedded_json)

    return ParsedPage(
        json_ld=json_ld,
        og_tags=og_tags,
        embedded_json=embedded_json,
        body_text=body_text,
        image_urls=image_urls,
        video_urls=video_urls,
        meta_tags=meta_tags,
        breadcrumbs=breadcrumbs,
        platform=platform,
        js_framework=js_framework,
        dom_variants=dom_variants,
        dom_sale_price=dom_sale_price,
        page_url=page_url,
        active_selections=active_selections,
        media_lookups=media_lookups,
    )


# ---------------------------------------------------------------------------
# Platform detection
# ---------------------------------------------------------------------------

_PLATFORM_RULES: list[tuple[str, list[re.Pattern]]] = [
    ("shopify", [
        re.compile(r"cdn\.shopify\.com", re.I),
        re.compile(r"Shopify\.", re.I),
        re.compile(r"shopify-section", re.I),
        re.compile(r"myshopify\.com", re.I),
        re.compile(r"ShopifyAnalytics", re.I),
        re.compile(r"shopify-features", re.I),
    ]),
    ("magento", [
        re.compile(r"Magento", re.I),
        re.compile(r"mage/", re.I),
        re.compile(r"catalogProductView", re.I),
    ]),
    ("salesforce_commerce_cloud", [
        re.compile(r"demandware", re.I),
        re.compile(r"sfcc", re.I),
        re.compile(r"Sites-", re.I),
    ]),
]

_JS_FRAMEWORK_RULES: list[tuple[str, list[re.Pattern]]] = [
    ("next.js", [re.compile(r"__NEXT_DATA__"), re.compile(r"_next/static")]),
    ("angular", [re.compile(r"ng-version", re.I)]),
    ("remix", [re.compile(r"__remixContext", re.I)]),
    ("svelte", [re.compile(r"__svelte", re.I)]),
    ("gatsby", [re.compile(r"___gatsby", re.I)]),
    ("vue", [re.compile(r"__vue__", re.I)]),
    ("nuxt", [re.compile(r"__NUXT__", re.I)]),
]


def _detect_platform(html: str) -> tuple[str, str]:
    """Detect e-commerce platform and JS framework from HTML content."""
    sample = html[:200_000]

    platform = "custom"
    for name, patterns in _PLATFORM_RULES:
        if any(p.search(sample) for p in patterns):
            platform = name
            break

    js_framework = "unknown"
    for name, patterns in _JS_FRAMEWORK_RULES:
        if any(p.search(sample) for p in patterns):
            js_framework = name
            break

    return platform, js_framework


# ---------------------------------------------------------------------------
# JSON-LD
# ---------------------------------------------------------------------------


def _extract_json_ld(soup: BeautifulSoup) -> list[dict]:
    """Extract all JSON-LD blocks from <script type="application/ld+json"> tags."""
    results: list[dict] = []
    for tag in soup.find_all("script", attrs={"type": "application/ld+json"}):
        text = tag.string
        if not text:
            continue
        try:
            data = json.loads(text)
            # Flatten arrays — some sites wrap JSON-LD in [...]
            if isinstance(data, list):
                results.extend(d for d in data if isinstance(d, dict))
            elif isinstance(data, dict):
                results.append(data)
        except (json.JSONDecodeError, TypeError):
            logger.debug("Skipping malformed JSON-LD block")
    return results


# ---------------------------------------------------------------------------
# Open Graph meta tags
# ---------------------------------------------------------------------------


def _extract_og_tags(soup: BeautifulSoup) -> dict[str, str]:
    """Extract Open Graph and product meta tags. Handles both property= and name= attributes."""
    tags: dict[str, str] = {}
    for meta in soup.find_all("meta"):
        prop = meta.get("property", "") or meta.get("name", "")
        if not isinstance(prop, str):
            continue
        content = meta.get("content", "")
        if not content:
            continue
        if prop.startswith("og:"):
            key = prop[3:]
            tags[key] = content
        elif prop.startswith("product:"):
            # Facebook product tags (e.g., product:price:amount -> price:amount)
            key = prop[8:]
            if key not in tags:
                tags[key] = content
    return tags


# ---------------------------------------------------------------------------
# Embedded JSON state objects
# ---------------------------------------------------------------------------


def _extract_embedded_json(soup: BeautifulSoup, html: str) -> dict[str, Any]:
    """Extract embedded JSON from script tags and window global assignments."""
    results: dict[str, Any] = {}

    # Pattern 1: <script type="application/json"|"text/json" id="...">
    for tag in soup.find_all("script"):
        tag_type = (tag.get("type") or "").lower()
        tag_id = tag.get("id")
        if tag_type in ("application/json", "text/json") and tag_id:
            text = tag.string
            if not text:
                continue
            try:
                results[tag_id] = json.loads(text)
            except (json.JSONDecodeError, TypeError):
                logger.debug(f"Skipping malformed JSON in script#{tag_id}")

    # Pattern 2: window.__VARIABLE__ = {...} assignments
    _extract_window_globals(soup, results)

    # Pattern 3: analytics tracking calls with product data (e.g., Shopify)
    _extract_analytics_tracking(soup, results)

    return results


def _extract_window_globals(soup: BeautifulSoup, results: dict[str, Any]) -> None:
    """Extract window.__X__ = {...} assignments from inline script tags."""
    pattern = re.compile(r"window\.(__[A-Z][A-Z0-9_]*__)\s*=\s*")

    for tag in soup.find_all("script"):
        # Skip scripts with src (external) or with a type that indicates structured data
        if tag.get("src") or tag.get("type") in ("application/json", "text/json", "application/ld+json"):
            continue
        text = tag.string
        if not text:
            continue

        for match in pattern.finditer(text):
            var_name = match.group(1)
            start = match.end()

            # Skip whitespace after =
            while start < len(text) and text[start] in " \t\n\r":
                start += 1

            if start >= len(text):
                continue

            # Only extract objects and arrays
            if text[start] not in ("{", "["):
                continue

            json_str = _brace_match(text, start)
            if not json_str:
                continue

            try:
                results[var_name] = json.loads(json_str)
            except (json.JSONDecodeError, TypeError):
                logger.debug(f"Skipping malformed JSON for {var_name}")


def _extract_analytics_tracking(soup: BeautifulSoup, results: dict[str, Any]) -> None:
    """Extract product data from analytics tracking calls (e.g., Shopify).

    Matches: .track("Viewed Product",{...})
    """
    pattern = re.compile(r'\.track\(\s*"Viewed Product"\s*,\s*')

    for tag in soup.find_all("script"):
        if tag.get("src") or tag.get("type") in ("application/json", "text/json", "application/ld+json"):
            continue
        text = tag.string
        if not text:
            continue

        match = pattern.search(text)
        if not match:
            continue

        json_str = _brace_match(text, match.end())
        if not json_str:
            continue

        try:
            data = json.loads(json_str)
            if isinstance(data, dict) and data.get("name"):
                results["_analytics_tracked_product"] = data
                return
        except (json.JSONDecodeError, TypeError):
            logger.debug("Skipping malformed analytics tracking JSON")


def _brace_match(text: str, start: int) -> str | None:
    """Extract a balanced JSON object/array from text starting at position start.

    Handles nested braces/brackets and string literals with escaped quotes.
    """
    if start >= len(text) or text[start] not in ("{", "["):
        return None

    depth = 0
    in_string = False
    escape_next = False
    i = start

    while i < len(text):
        c = text[i]

        if escape_next:
            escape_next = False
            i += 1
            continue

        if c == "\\" and in_string:
            escape_next = True
            i += 1
            continue

        if c == '"':
            in_string = not in_string
            i += 1
            continue

        if in_string:
            i += 1
            continue

        # Outside strings: track brace/bracket depth
        if c in ("{", "["):
            depth += 1
        elif c in ("}", "]"):
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

        i += 1

    return None


# ---------------------------------------------------------------------------
# Body text extraction
# ---------------------------------------------------------------------------


def _extract_body_text(soup: BeautifulSoup) -> str:
    """Extract visible text from the page body, stripping noise elements."""
    body = soup.find("body")
    if not body:
        body = soup

    # Work on a copy so we don't mutate the original
    body_copy = BeautifulSoup(str(body), "lxml")

    # Remove noise elements
    for tag_name in ("script", "style", "noscript", "svg", "nav", "header", "footer", "iframe"):
        for el in body_copy.find_all(tag_name):
            el.decompose()

    text = body_copy.get_text(separator=" ", strip=True)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Truncate to keep LLM context manageable
    return text[:5000]


# ---------------------------------------------------------------------------
# Image URL extraction
# ---------------------------------------------------------------------------

# Cross-sell / recommendation section class/id patterns
_CROSS_SELL_RE = re.compile(
    r"cross[-_]?sell|recommend|related[-_]?product|you[-_]?may[-_]?also|"
    r"also[-_]?like|suggested|complementary|upsell|frequently[-_]?bought|"
    r"people[-_]?also|similar[-_]?item|pair[-_]?with|shop[-_]?the[-_]?look|"
    r"complete[-_]?the[-_]?look|bought[-_]?together",
    re.IGNORECASE,
)


def _find_non_product_images(soup: BeautifulSoup) -> set[int]:
    """Build a set of <img> element IDs that are in non-product sections.

    Identifies images inside navigation, headers, footers, hidden elements,
    and cross-sell/recommendation modules to prevent contamination from
    non-product areas of the page.
    """
    excluded: set[int] = set()

    # 1. Images inside <nav>, <header>, <footer>
    for tag_name in ("nav", "header", "footer"):
        for section in soup.find_all(tag_name):
            for img in section.find_all("img"):
                excluded.add(id(img))

    # 2. Images inside hidden elements (class="hide" or class="hidden")
    for cls in ("hide", "hidden"):
        for el in soup.find_all(class_=cls):
            for img in el.find_all("img"):
                excluded.add(id(img))

    # 3. Images inside display:none / visibility:hidden elements
    for el in soup.find_all(style=re.compile(r"display\s*:\s*none|visibility\s*:\s*hidden", re.I)):
        for img in el.find_all("img"):
            excluded.add(id(img))

    # 4. Images in cross-sell / recommendation sections
    for el in soup.find_all(attrs={"class": _CROSS_SELL_RE}):
        for img in el.find_all("img"):
            excluded.add(id(img))
    for el in soup.find_all(attrs={"id": _CROSS_SELL_RE}):
        for img in el.find_all("img"):
            excluded.add(id(img))

    return excluded


def _extract_image_urls(soup: BeautifulSoup) -> list[str]:
    """Extract product image URLs from <img> tags.

    Excludes images from non-product areas (navigation, hidden sections,
    cross-sell modules) to prevent contamination.

    Checks srcset/data-srcset first (preferring the highest-resolution variant),
    then falls back to src/data-src.
    """
    excluded = _find_non_product_images(soup)

    urls: list[str] = []
    for img in soup.find_all("img"):
        if id(img) in excluded:
            continue

        # Prefer the highest-resolution URL from srcset when available
        best_srcset = _best_from_srcset(img.get("srcset") or img.get("data-srcset"))
        if best_srcset:
            best_srcset = _normalize_url(best_srcset)
            if best_srcset and _looks_like_image_url(best_srcset):
                urls.append(best_srcset)
                continue  # srcset found a good URL, skip src/data-src for this tag

        for attr in ("src", "data-src"):
            url = img.get(attr)
            if url and isinstance(url, str):
                url = _normalize_url(url)
                if url and _looks_like_image_url(url):
                    urls.append(url)
    return list(dict.fromkeys(urls))  # deduplicate preserving order


def _best_from_srcset(srcset: str | None) -> str | None:
    """Parse an srcset attribute and return the highest-resolution URL.

    Handles both width descriptors (e.g. '800w') and pixel-density
    descriptors (e.g. '2x').  Falls back to the last entry when no
    descriptor is present.
    """
    if not srcset or not isinstance(srcset, str):
        return None

    best_url: str | None = None
    best_value: float = 0

    for entry in srcset.split(","):
        parts = entry.strip().split()
        if not parts:
            continue
        url = parts[0]
        if not url:
            continue

        if len(parts) >= 2:
            descriptor = parts[-1].strip().lower()
            try:
                if descriptor.endswith("w") or descriptor.endswith("x"):
                    value = float(descriptor[:-1])
                else:
                    value = 0
            except ValueError:
                value = 0
        else:
            # No descriptor — treat as a single candidate
            value = 1

        if value >= best_value:
            best_url = url
            best_value = value

    return best_url


def _normalize_url(url: str) -> str:
    """Normalize a URL: add https: to protocol-relative URLs."""
    url = url.strip()
    if url.startswith("//"):
        return "https:" + url
    return url


def _looks_like_image_url(url: str) -> bool:
    """Basic check that a URL looks like a product image (not a tracking pixel or icon)."""
    if not url.startswith("http"):
        return False
    # Skip common non-product patterns
    skip_patterns = ("favicon", "logo", "pixel", "tracking", "analytics", "1x1", "spacer", "flyout")
    url_lower = url.lower()
    return not any(p in url_lower for p in skip_patterns)


# ---------------------------------------------------------------------------
# Video URL extraction
# ---------------------------------------------------------------------------


def _extract_video_urls(html: str) -> list[str]:
    """Extract video URLs (.mp4, .webm) from the HTML."""
    pattern = re.compile(r'https?://[^\s"\'<>]+\.(?:mp4|webm)', re.IGNORECASE)
    urls = pattern.findall(html)
    return list(dict.fromkeys(urls))  # deduplicate


# ---------------------------------------------------------------------------
# Standard meta tags
# ---------------------------------------------------------------------------


def _extract_meta_tags(soup: BeautifulSoup) -> dict[str, str]:
    """Extract standard meta tags (description, keywords, etc.)."""
    tags: dict[str, str] = {}
    for name in ("description", "keywords", "title"):
        meta = soup.find("meta", attrs={"name": name})
        if meta and meta.get("content"):
            tags[name] = meta["content"]
    return tags


# ---------------------------------------------------------------------------
# Breadcrumb extraction
# ---------------------------------------------------------------------------


def _extract_breadcrumbs(json_ld: list[dict], soup: BeautifulSoup) -> list[str]:
    """Extract breadcrumb trail from JSON-LD, microdata, or HTML nav elements.

    Returns ordered list of breadcrumb names.
    """
    # Try JSON-LD BreadcrumbList first
    for block in json_ld:
        if block.get("@type") == "BreadcrumbList":
            items = block.get("itemListElement", [])
            if isinstance(items, list):
                sorted_items = sorted(items, key=lambda x: x.get("position", 0))
                names = [str(item.get("name", "")) for item in sorted_items if item.get("name")]
                if names:
                    return names

    # Fallback: microdata BreadcrumbList
    bc_list = soup.find(itemtype=re.compile(r"schema\.org/BreadcrumbList"))
    if bc_list:
        items = bc_list.find_all(itemtype=re.compile(r"schema\.org/ListItem"))
        positioned = []
        for item in items:
            pos_tag = item.find("meta", attrs={"itemprop": "position"})
            pos = int(pos_tag["content"]) if pos_tag and pos_tag.get("content") else 999
            name_tag = item.find(attrs={"itemprop": "name"})
            name = name_tag.get_text(strip=True) if name_tag else ""
            if name:
                positioned.append((pos, name))
        if positioned:
            positioned.sort(key=lambda x: x[0])
            return [name for _, name in positioned]

    # Fallback: HTML nav breadcrumbs
    return _extract_html_breadcrumbs(soup)


_BC_NOISE = {"home", ">", "/", "you are here:", "you are here", ""}


def _extract_html_breadcrumbs(soup: BeautifulSoup) -> list[str]:
    """Extract breadcrumbs from HTML nav elements and breadcrumb-classed containers."""
    container = None

    # Pattern 1: <nav aria-label="breadcrumb(s)">
    container = soup.find("nav", attrs={"aria-label": re.compile(r"breadcrumb", re.I)})

    # Pattern 2: element with class containing "breadcrumb"
    if not container:
        container = soup.find(attrs={"class": re.compile(r"breadcrumb", re.I)})

    # Pattern 3: element with id containing "breadcrumb"
    if not container:
        container = soup.find(attrs={"id": re.compile(r"breadcrumb", re.I)})

    if not container:
        return []

    # Try <a> tags first (most structured)
    names = []
    links = container.find_all("a")
    if links:
        for link in links:
            text = link.get_text(strip=True)
            if text and text.lower() not in _BC_NOISE:
                names.append(text)
        if names:
            return names

    # Fallback: <li> text
    items = container.find_all("li")
    if items:
        for item in items:
            text = item.get_text(strip=True)
            if text and text.lower() not in _BC_NOISE and len(text) > 1 and text not in names:
                names.append(text)
        if names:
            return names

    # Fallback: <span> text
    spans = container.find_all("span")
    for span in spans:
        text = span.get_text(strip=True)
        if text and text.lower() not in _BC_NOISE and len(text) > 1 and text not in names:
            names.append(text)

    return names


# ---------------------------------------------------------------------------
# DOM-based variant extraction
# ---------------------------------------------------------------------------

_SWATCH_SELECTORS = [
    {"class": re.compile(r"swatch", re.I)},
    {"class": re.compile(r"color-option", re.I)},
    {"class": re.compile(r"colour-option", re.I)},
    {"class": re.compile(r"color-selector", re.I)},
    {"class": re.compile(r"product-colors", re.I)},
]

_SIZE_SELECTORS = [
    {"class": re.compile(r"size-option", re.I)},
    {"class": re.compile(r"size-selector", re.I)},
    {"class": re.compile(r"size-button", re.I)},
    {"class": re.compile(r"product-size", re.I)},
]


def _extract_dom_variants(soup: BeautifulSoup) -> list[dict]:
    """Extract variant signals from DOM elements (select, swatch, button).

    Returns list of dicts with keys: type (size/color), values (list[str]), source.
    """
    results: list[dict] = []

    # Pattern 1: <select> dropdowns with size/color options
    # Only check the element's own attributes (name, id, aria-label, class) —
    # NOT str(select) which includes child option text and causes false positives
    # (e.g., "Colorado" matching "color").
    for select in soup.find_all("select"):
        attrs_str = " ".join(
            str(select.get(a, "")) for a in ("name", "id", "aria-label", "class")
        ).lower()

        var_type = None
        if any(s in attrs_str for s in ("size", "dimension", "length")):
            var_type = "size"
        elif any(s in attrs_str for s in ("color", "colour")):
            var_type = "color"

        if not var_type:
            continue

        values = []
        for opt in select.find_all("option"):
            text = opt.get_text(strip=True)
            val = opt.get("value", "")
            if text.lower() in ("select", "choose", "pick", "--", "", "select size", "select color", "choose size", "choose color"):
                continue
            if text:
                values.append(text)
            elif val and val not in ("", "0"):
                values.append(val)

        if values:
            results.append({"type": var_type, "values": values, "source": "select"})

    # Pattern 2: Color swatch containers
    for selector in _SWATCH_SELECTORS:
        container = soup.find(attrs=selector)
        if not container:
            continue
        color_values = []
        for el in container.find_all(["button", "a", "label", "span", "li", "div"]):
            color = (
                el.get("data-color")
                or el.get("data-value")
                or el.get("aria-label")
                or el.get("title")
            )
            if not color:
                text = el.get_text(strip=True)
                if text and len(text) < 50:
                    color = text
            if color and len(color) < 50 and color not in color_values:
                color_values.append(color)
        if color_values:
            results.append({"type": "color", "values": color_values, "source": "swatch"})
            break

    # Pattern 3: Size button groups
    for selector in _SIZE_SELECTORS:
        container = soup.find(attrs=selector)
        if not container:
            continue
        size_values = []
        for el in container.find_all(["button", "a", "label", "span", "li"]):
            size = (
                el.get("data-size")
                or el.get("data-value")
                or el.get_text(strip=True)
            )
            if size and len(size) < 20 and size not in size_values:
                size_values.append(size)
        if size_values:
            results.append({"type": "size", "values": size_values, "source": "button"})
            break

    return results


# ---------------------------------------------------------------------------
# DOM-based sale price detection
# ---------------------------------------------------------------------------

_PRICE_RE = re.compile(r"\$(\d+(?:,\d{3})*(?:\.\d{2})?)")

_COMPARE_PRICE_SELECTORS = [
    {"class": re.compile(r"compare.?at.?price", re.I)},
    {"class": re.compile(r"original.?price", re.I)},
    {"class": re.compile(r"was.?price", re.I)},
    {"class": re.compile(r"price--compare", re.I)},
    {"class": re.compile(r"price--was", re.I)},
    {"class": re.compile(r"msrp", re.I)},
]

_SALE_PRICE_SELECTORS = [
    {"class": re.compile(r"sale.?price", re.I)},
    {"class": re.compile(r"current.?price", re.I)},
    {"class": re.compile(r"price--sale", re.I)},
    {"class": re.compile(r"price--current", re.I)},
    {"class": re.compile(r"special.?price", re.I)},
]


def _extract_dom_sale_price(soup: BeautifulSoup) -> dict[str, Any]:
    """Detect sale price patterns from CSS classes, line-through styles, and strikethrough tags."""
    result: dict[str, Any] = {"has_sale": False, "original_price": None, "sale_price": None}

    # Pattern 1: CSS class-based compare/original/was price
    for selector in _COMPARE_PRICE_SELECTORS:
        el = soup.find(attrs=selector)
        if el:
            text = el.get_text(strip=True)
            match = _PRICE_RE.search(text)
            if match:
                result["has_sale"] = True
                result["original_price"] = float(match.group(1).replace(",", ""))
                break

    # Pattern 2: Inline style line-through
    if not result["has_sale"]:
        for el in soup.find_all(style=re.compile(r"line-through", re.I)):
            text = el.get_text(strip=True)
            match = _PRICE_RE.search(text)
            if match:
                result["has_sale"] = True
                result["original_price"] = float(match.group(1).replace(",", ""))
                break

    # Pattern 3: <s>, <strike>, <del> wrapping prices
    if not result["has_sale"]:
        for tag_name in ("s", "strike", "del"):
            for el in soup.find_all(tag_name):
                text = el.get_text(strip=True)
                match = _PRICE_RE.search(text)
                if match:
                    result["has_sale"] = True
                    result["original_price"] = float(match.group(1).replace(",", ""))
                    break
            if result["has_sale"]:
                break

    # If original price found, try to find the sale/current price nearby
    if result["has_sale"]:
        for selector in _SALE_PRICE_SELECTORS:
            el = soup.find(attrs=selector)
            if el:
                text = el.get_text(strip=True)
                match = _PRICE_RE.search(text)
                if match:
                    price_val = float(match.group(1).replace(",", ""))
                    if result["original_price"] and price_val < result["original_price"]:
                        result["sale_price"] = price_val
                    break

    return result


# ---------------------------------------------------------------------------
# Page URL extraction
# ---------------------------------------------------------------------------


def _extract_page_url(soup: BeautifulSoup, og_tags: dict[str, str]) -> str:
    """Extract the page URL from canonical link or og:url meta tag."""
    canonical = soup.find("link", rel="canonical")
    if canonical and canonical.get("href"):
        return str(canonical["href"])
    return og_tags.get("url", "")


# ---------------------------------------------------------------------------
# Active selection detection
# ---------------------------------------------------------------------------

# Keys that indicate a "selected" flag on an option/answer
_SELECTED_FLAG_KEYS = re.compile(
    r"(?i)^(isSelected|selected|is_selected|active|isActive|is_active|checked)$"
)
# Keys that give the dimension name on a question/option-group
_DIMENSION_TYPE_KEYS = re.compile(
    r"(?i)^(type|name|label|question|dimension|attribute|optionType)$"
)
# Keys that give the human-readable value of an option/answer
_ANSWER_LABEL_KEYS = re.compile(
    r"(?i)^(title|name|value|label|text|displayValue|displayName)$"
)
# Keys that contain the array of options/answers
_ANSWERS_ARRAY_KEYS = re.compile(
    r"(?i)^(answers|options|values|items|choices|swatches)$"
)
# Keys that directly name a selected variant value
_VARIANT_NAME_KEYS = re.compile(
    r"(?i)^(variantName|variant_name|selectedColor|selectedColour|"
    r"selectedFinish|selectedMaterial|selectedPattern|selectedStyle|"
    r"selectedVariant|activeColor|activeVariant|currentColor|currentVariant)$"
)
# Dimension hints from key names
_DIMENSION_HINT_MAP: list[tuple[re.Pattern, str]] = [
    (re.compile(r"(?i)color|colour"), "color"),
    (re.compile(r"(?i)finish"), "finish"),
    (re.compile(r"(?i)material"), "material"),
    (re.compile(r"(?i)pattern"), "pattern"),
    (re.compile(r"(?i)style"), "style"),
    (re.compile(r"(?i)size"), "size"),
]


def _extract_active_selections(
    soup: BeautifulSoup,
    embedded_json: dict[str, Any],
) -> list[dict]:
    """Detect which dimension values are currently selected on this page.

    Returns list of {"dimension": str, "value": str, "source": str}.
    """
    results: list[dict] = []
    seen: set[tuple[str, str]] = set()  # (dimension, value) dedup

    def _add(dimension: str, value: str, source: str) -> None:
        key = (dimension.lower(), value)
        if key not in seen:
            seen.add(key)
            results.append({"dimension": dimension.lower(), "value": value, "source": source})

    # Signal 1: isSelected flags in question/answer structures
    _detect_json_selected_flags(embedded_json, _add)

    # Signal 2: variantName / selectedColor / etc. fields
    _detect_json_variant_name(embedded_json, _add)

    # Signal 3: DOM selected state
    _detect_dom_selected(soup, _add)

    return results


def _detect_json_selected_flags(
    data: Any,
    add_fn: Any,
    depth: int = 0,
) -> None:
    """Find isSelected flags in question/answer-style structures.

    Detects patterns like:
      {type: "COLOR", answers: [{title: "Black", isSelected: true}, ...]}
    """
    if depth > 10 or not isinstance(data, (dict, list)):
        return

    if isinstance(data, dict):
        # Check if this dict has an answers/options array with selected flags
        dimension = None
        for k, v in data.items():
            if _DIMENSION_TYPE_KEYS.match(k) and isinstance(v, str) and v.strip():
                dimension = v.strip().lower()
                break

        if dimension:
            for k, v in data.items():
                if not _ANSWERS_ARRAY_KEYS.match(k) or not isinstance(v, list):
                    continue
                for item in v:
                    if not isinstance(item, dict):
                        continue
                    # Check for selected flag
                    is_selected = False
                    label = None
                    for ik, iv in item.items():
                        if _SELECTED_FLAG_KEYS.match(ik) and iv is True:
                            is_selected = True
                        elif _ANSWER_LABEL_KEYS.match(ik) and isinstance(iv, str) and iv.strip():
                            label = iv.strip()
                    if is_selected and label:
                        add_fn(dimension, label, "json_selected")

        # Recurse into all values
        for v in data.values():
            _detect_json_selected_flags(v, add_fn, depth + 1)

    elif isinstance(data, list):
        for item in data:
            _detect_json_selected_flags(item, add_fn, depth + 1)


_PRODUCT_SIGNAL_KEYS_PARSER = {
    "name", "title", "price", "description", "brand", "brandName", "sku",
    "image", "images", "media", "variants", "items",
}


def _detect_json_variant_name(
    data: Any,
    add_fn: Any,
    depth: int = 0,
    _found: list | None = None,
) -> None:
    """Find variantName/selectedColor/etc. fields in embedded JSON.

    Only picks up from the first product-level object found (with 3+ product
    signal keys) to avoid false positives from sibling/related product arrays.
    """
    if _found is None:
        _found = []
    if depth > 10 or _found or not isinstance(data, (dict, list)):
        return

    if isinstance(data, dict):
        is_product_obj = len(_PRODUCT_SIGNAL_KEYS_PARSER & set(data.keys())) >= 3
        if is_product_obj:
            for k, v in data.items():
                if not _VARIANT_NAME_KEYS.match(k) or not isinstance(v, str) or not v.strip():
                    continue
                value = v.strip()
                dimension = _infer_dimension_from_key(k)
                if dimension == "variant":
                    dimension = _infer_dimension_from_siblings(data) or "variant"
                add_fn(dimension, value, "json_variant_name")
                _found.append(True)
                return

        for v in data.values():
            _detect_json_variant_name(v, add_fn, depth + 1, _found)

    elif isinstance(data, list):
        for item in data:
            _detect_json_variant_name(item, add_fn, depth + 1, _found)


def _infer_dimension_from_key(key: str) -> str:
    """Infer dimension type from a key name like 'selectedColor' -> 'color'."""
    for pattern, dimension in _DIMENSION_HINT_MAP:
        if pattern.search(key):
            return dimension
    return "variant"


def _infer_dimension_from_siblings(obj: dict) -> str | None:
    """Infer dimension from sibling keys like color_swatch, finish_group, etc."""
    for k in obj:
        if not isinstance(k, str):
            continue
        for pattern, dimension in _DIMENSION_HINT_MAP:
            if pattern.search(k) and k.lower() not in ("color", "colour"):
                return dimension
    return None


def _detect_dom_selected(soup: BeautifulSoup, add_fn: Any) -> None:
    """Detect selected variant options from DOM state."""
    all_selectors = _SWATCH_SELECTORS + _SIZE_SELECTORS
    for selector in all_selectors:
        container = soup.find(attrs=selector)
        if not container:
            continue

        # Determine dimension from container
        container_str = " ".join(
            str(container.get(a, "")) for a in ("class", "id", "aria-label")
        ).lower()
        dimension = None
        for pattern, dim in _DIMENSION_HINT_MAP:
            if pattern.search(container_str):
                dimension = dim
                break
        if not dimension:
            continue

        for el in container.find_all(
            ["button", "a", "label", "span", "li", "div", "option"]
        ):
            is_selected = (
                el.get("aria-checked") == "true"
                or el.get("aria-selected") == "true"
                or "selected" in (el.get("class") or [])
                or "active" in (el.get("class") or [])
                or el.get("data-selected") == "true"
            )
            if not is_selected:
                continue
            label = (
                el.get("data-color")
                or el.get("data-value")
                or el.get("data-size")
                or el.get("aria-label")
                or el.get("title")
                or el.get_text(strip=True)
            )
            if label and len(label) < 80:
                add_fn(dimension, label, "dom_selected")


# ---------------------------------------------------------------------------
# Media lookup extraction
# ---------------------------------------------------------------------------

_MEDIA_ID_KEYS = re.compile(r"(?i)^(id|mediaId|media_id)$")
_MEDIA_SRC_KEYS = re.compile(r"(?i)^(src|url|source|imageUrl|image_url|href)$")


def _extract_media_lookups(embedded_json: dict[str, Any]) -> list[list[dict]]:
    """Find arrays of {id, src} media objects in embedded JSON.

    Generically detects arrays where items have both an ID-like key and
    a src/URL-like key. Normalizes to [{id: str, src: str}, ...].
    """
    results: list[list[dict]] = []
    _find_media_arrays(embedded_json, results)
    return results


def _find_media_arrays(
    data: Any,
    results: list[list[dict]],
    depth: int = 0,
) -> None:
    """Recursively find media lookup arrays."""
    if depth > 10:
        return
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list) and len(v) >= 2 and isinstance(v[0], dict):
                normalized = _try_normalize_media_array(v)
                if normalized:
                    results.append(normalized)
                else:
                    _find_media_arrays(v, results, depth + 1)
            else:
                _find_media_arrays(v, results, depth + 1)
    elif isinstance(data, list):
        for item in data:
            _find_media_arrays(item, results, depth + 1)


def _try_normalize_media_array(arr: list[dict]) -> list[dict] | None:
    """Check if array is a media lookup table and normalize to [{id, src}]."""
    sample = arr[0]
    if not isinstance(sample, dict):
        return None

    id_key = None
    src_key = None
    for k in sample:
        if id_key is None and _MEDIA_ID_KEYS.match(k):
            id_key = k
        elif src_key is None and _MEDIA_SRC_KEYS.match(k) and isinstance(sample[k], str):
            src_key = k
    if not id_key or not src_key:
        return None

    # Validate: at least half the items have both fields
    valid = sum(1 for item in arr if isinstance(item, dict) and item.get(id_key) and item.get(src_key))
    if valid < len(arr) * 0.5:
        return None

    return [
        {"id": str(item[id_key]), "src": str(item[src_key])}
        for item in arr
        if isinstance(item, dict) and item.get(id_key) and item.get(src_key)
    ]
