"""
Product extraction orchestrator.

Processes all HTML product pages in parallel using asyncio.gather,
running each through: parse -> hydrate -> LLM fill -> validate.
"""

import asyncio
import json
import logging
import time
from pathlib import Path

from models import Product
from parser import parse_html
from extractor import extract_product, ExtractionMetrics, PRODUCT_FIELDS

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"
OUTPUT_FILE = Path(__file__).parent / "products.json"


async def process_file(filepath: Path) -> tuple[Product, ExtractionMetrics, float]:
    """Process a single HTML file through the full extraction pipeline.

    Returns (product, extraction_metrics, parse_time_seconds).
    """
    logger.info(f"Processing {filepath.name}...")

    html = filepath.read_text(encoding="utf-8")

    # Step 1: Parse structured data from HTML (zero cost)
    t0 = time.monotonic()
    parsed = parse_html(html)
    parse_time = time.monotonic() - t0

    logger.info(
        f"  Parsed: {len(parsed.json_ld)} JSON-LD, "
        f"{len(parsed.og_tags)} OG tags, "
        f"{len(parsed.embedded_json)} embedded JSON, "
        f"{len(parsed.image_urls)} images, "
        f"{len(parsed.video_urls)} videos"
    )

    # Step 2-4: Extract, fill gaps with LLM, validate
    product, metrics = await extract_product(parsed, filepath.name)
    logger.info(
        f"  Result: {product.name} ({product.brand}) - "
        f"{product.price.price} {product.price.currency} | "
        f"Category: {product.category.name} | "
        f"Features: {len(product.key_features)}, "
        f"Images: {len(product.image_urls)}, "
        f"Variants: {len(product.variants)}, "
        f"Colors: {len(product.colors)}"
    )

    return product, metrics, parse_time


async def process_all() -> tuple[list[Product], list[ExtractionMetrics], list[float], int]:
    """Process all HTML files in the data directory concurrently.

    Returns (products, metrics_list, parse_times, failure_count).
    """
    html_files = sorted(DATA_DIR.glob("*.html"))
    logger.info(f"Found {len(html_files)} HTML files to process")

    # Process all files in parallel
    results = await asyncio.gather(
        *[process_file(f) for f in html_files],
        return_exceptions=True,
    )

    # Separate successes from failures
    products: list[Product] = []
    all_metrics: list[ExtractionMetrics] = []
    parse_times: list[float] = []
    failures = 0

    for filepath, result in zip(html_files, results):
        if isinstance(result, Exception):
            logger.error(f"Failed to process {filepath.name}: {result}", exc_info=result)
            failures += 1
        else:
            product, metrics, parse_time = result
            products.append(product)
            all_metrics.append(metrics)
            parse_times.append(parse_time)

    return products, all_metrics, parse_times, failures


def print_report(
    products: list[Product],
    all_metrics: list[ExtractionMetrics],
    parse_times: list[float],
    failures: int,
    wall_clock: float,
) -> None:
    """Print a comprehensive extraction report."""
    total_files = len(all_metrics) + failures
    n = len(all_metrics)

    # ── Reliability ──────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("EXTRACTION REPORT")
    print(f"{'='*70}")

    print(f"\n── Reliability ──")
    print(f"  Files attempted:  {total_files}")
    print(f"  Succeeded:        {n}")
    print(f"  Failed:           {failures}")
    print(f"  Success rate:     {n/total_files*100:.0f}%" if total_files else "  N/A")

    if not all_metrics:
        print("\n  No successful extractions to report on.")
        return

    # ── Parser vs LLM (cost lever) ──────────────────────────────────
    print(f"\n── Parser vs LLM (free vs paid) ──")
    total_fields_possible = n * len(PRODUCT_FIELDS)
    total_from_parser = sum(len(m.fields_from_parser) for m in all_metrics)
    total_from_llm = sum(len(m.fields_from_llm) for m in all_metrics)
    total_missing = sum(len(m.fields_missing_after_all) for m in all_metrics)

    print(f"  Fields filled by parser (free):  {total_from_parser}/{total_fields_possible} "
          f"({total_from_parser/total_fields_possible*100:.0f}%)")
    print(f"  Fields filled by LLM (paid):     {total_from_llm}/{total_fields_possible} "
          f"({total_from_llm/total_fields_possible*100:.0f}%)")
    print(f"  Fields still empty after all:     {total_missing}/{total_fields_possible} "
          f"({total_missing/total_fields_possible*100:.0f}%)")

    llm_skipped = sum(1 for m in all_metrics if m.llm_skipped)
    print(f"  Products needing zero LLM calls:  {llm_skipped}/{n}")

    # Per-field breakdown
    print(f"\n  Per-field source breakdown:")
    print(f"  {'Field':<20} {'Parser':>8} {'LLM':>8} {'Empty':>8}")
    print(f"  {'-'*46}")
    for field in PRODUCT_FIELDS:
        from_parser = sum(1 for m in all_metrics if field in m.fields_from_parser)
        from_llm = sum(1 for m in all_metrics if field in m.fields_from_llm)
        empty = sum(1 for m in all_metrics if field in m.fields_missing_after_all)
        print(f"  {field:<20} {from_parser:>7}  {from_llm:>7}  {empty:>7}")

    # ── LLM Cost ────────────────────────────────────────────────────
    print(f"\n── LLM Usage ──")
    total_llm_calls = sum(m.llm_calls for m in all_metrics)
    print(f"  Total LLM calls:  {total_llm_calls} across {n} products")
    print(f"  Avg calls/product: {total_llm_calls/n:.1f}")
    for m in all_metrics:
        label = "skipped" if m.llm_skipped else f"{m.llm_calls} call(s)"
        print(f"    {m.filename:<25} {label}")

    # ── Category Resolution ─────────────────────────────────────────
    print(f"\n── Category Resolution ──")
    resolutions: dict[str, int] = {}
    for m in all_metrics:
        resolutions[m.category_resolution] = resolutions.get(m.category_resolution, 0) + 1
    for method, count in sorted(resolutions.items(), key=lambda x: -x[1]):
        label = {
            "parser": "From structured data (free)",
            "llm_exact": "LLM exact match (1 call)",
            "fuzzy_match": "LLM + fuzzy correction (1 call, no retry)",
            "llm_retry": "LLM retry needed (2 calls)",
            "failed": "Failed",
        }.get(method, method)
        print(f"  {label:<45} {count}/{n}")
    print()
    for m in all_metrics:
        print(f"    {m.filename:<25} {m.category_resolution}")

    # ── Output Richness ─────────────────────────────────────────────
    print(f"\n── Output Richness ──")
    print(f"  {'File':<25} {'Variants':>9} {'Images':>8} {'Features':>9} {'Colors':>8} {'Video':>7} {'Sale':>6}")
    print(f"  {'-'*73}")
    for m in all_metrics:
        print(f"  {m.filename:<25} {m.num_variants:>9} {m.num_images:>8} "
              f"{m.num_features:>9} {m.num_colors:>8} "
              f"{'yes' if m.has_video else '-':>7} "
              f"{'yes' if m.has_sale_price else '-':>6}")
    # Totals
    print(f"  {'-'*73}")
    print(f"  {'TOTAL':<25} {sum(m.num_variants for m in all_metrics):>9} "
          f"{sum(m.num_images for m in all_metrics):>8} "
          f"{sum(m.num_features for m in all_metrics):>9} "
          f"{sum(m.num_colors for m in all_metrics):>8} "
          f"{sum(1 for m in all_metrics if m.has_video):>7} "
          f"{sum(1 for m in all_metrics if m.has_sale_price):>6}")

    # ── Timing ──────────────────────────────────────────────────────
    print(f"\n── Timing ──")
    print(f"  Wall clock (total):  {wall_clock:.2f}s")
    print(f"  {'File':<25} {'Parse':>8} {'Hydrate':>9} {'LLM':>8} {'Validate':>10} {'Total':>8}")
    print(f"  {'-'*69}")
    for m, pt in zip(all_metrics, parse_times):
        print(f"  {m.filename:<25} {pt:>7.3f}s {m.hydrate_time:>8.3f}s "
              f"{m.llm_fill_time:>7.3f}s {m.validate_time:>9.3f}s {m.total_time:>7.3f}s")

    total_parse = sum(parse_times)
    total_hydrate = sum(m.hydrate_time for m in all_metrics)
    total_llm_time = sum(m.llm_fill_time for m in all_metrics)
    total_validate = sum(m.validate_time for m in all_metrics)
    sum_total = sum(m.total_time for m in all_metrics)

    print(f"  {'-'*69}")
    print(f"  {'SUM':<25} {total_parse:>7.3f}s {total_hydrate:>8.3f}s "
          f"{total_llm_time:>7.3f}s {total_validate:>9.3f}s {sum_total:>7.3f}s")

    if sum_total > 0:
        pct_free = (total_parse + total_hydrate) / sum_total * 100
        pct_llm = total_llm_time / sum_total * 100
        print(f"\n  Time in free stages (parse+hydrate): {pct_free:.1f}%")
        print(f"  Time in LLM stages (fill+retry):     {pct_llm:.1f}%")

    # ── Scale Projections ───────────────────────────────────────────
    print(f"\n── Scale Projections ──")
    avg_llm_calls = total_llm_calls / n
    avg_time = wall_clock / n  # wall clock per product (with parallelism)
    seq_time = sum_total / n   # sequential time per product
    print(f"  Avg LLM calls per product:   {avg_llm_calls:.1f}")
    print(f"  Avg time per product (seq):  {seq_time:.2f}s")
    print(f"  Avg time per product (wall): {avg_time:.2f}s")
    print(f"  Est. 1K products (wall):     {avg_time * 1000:.0f}s ({avg_time * 1000 / 60:.1f}min)")
    print(f"  Est. 1K products (seq):      {seq_time * 1000:.0f}s ({seq_time * 1000 / 60:.1f}min)")

    print(f"\n{'='*70}")


async def main() -> None:
    t_wall_start = time.monotonic()
    products, all_metrics, parse_times, failures = await process_all()
    wall_clock = time.monotonic() - t_wall_start

    # Print per-product summary
    print(f"\n{'='*60}")
    print(f"Extracted {len(products)} products:")
    print(f"{'='*60}")

    for p in products:
        print(f"\n  {p.name}")
        print(f"    Brand:    {p.brand}")
        print(f"    Price:    {p.price.price} {p.price.currency}", end="")
        if p.price.compare_at_price:
            print(f" (was {p.price.compare_at_price})", end="")
        print()
        print(f"    Category: {p.category.name}")
        print(f"    Features: {len(p.key_features)} items")
        print(f"    Images:   {len(p.image_urls)} URLs")
        print(f"    Variants: {len(p.variants)} configurations")
        print(f"    Colors:   {p.colors}")
        if p.video_url:
            print(f"    Video:    {p.video_url}")

    # Write products to JSON file
    products_json = [p.model_dump() for p in products]
    OUTPUT_FILE.write_text(json.dumps(products_json, indent=2))
    logger.info(f"\nWrote {len(products)} products to {OUTPUT_FILE}")

    # Print the full report
    print_report(products, all_metrics, parse_times, failures, wall_clock)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    asyncio.run(main())
