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


def parse_html(html: str) -> ParsedPage:
    """Parse an HTML page and extract all structured data sources."""
    soup = BeautifulSoup(html, "lxml")

    json_ld = _extract_json_ld(soup)
    og_tags = _extract_og_tags(soup)
    embedded_json = _extract_embedded_json(soup, html)
    body_text = _extract_body_text(soup)
    meta_tags = _extract_meta_tags(soup)
    image_urls = _extract_image_urls(soup)
    video_urls = _extract_video_urls(html)
    breadcrumbs = _extract_breadcrumbs(json_ld, soup)

    return ParsedPage(
        json_ld=json_ld,
        og_tags=og_tags,
        embedded_json=embedded_json,
        body_text=body_text,
        image_urls=image_urls,
        video_urls=video_urls,
        meta_tags=meta_tags,
        breadcrumbs=breadcrumbs,
    )


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


def _extract_image_urls(soup: BeautifulSoup) -> list[str]:
    """Extract all image URLs from <img> tags.

    Checks srcset/data-srcset first (preferring the highest-resolution variant),
    then falls back to src/data-src.
    """
    urls: list[str] = []
    for img in soup.find_all("img"):
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
    skip_patterns = ("favicon", "logo", "pixel", "tracking", "analytics", "1x1", "spacer")
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
    """Extract breadcrumb trail from JSON-LD BreadcrumbList or microdata.

    Returns ordered list of breadcrumb names (excluding the site root and product name).
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
        # Sort by position if available
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

    return []
