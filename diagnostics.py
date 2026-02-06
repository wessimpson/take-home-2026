"""
Diagnostic: run parser + programmatic hydration only (no LLM).
Reports what fields are filled vs missing for each HTML file.
"""

from pathlib import Path

from extractor import _hydrate_fields
from models import Price, Variant
from parser import parse_html

DATA_DIR = Path(__file__).parent / "data"
REQUIRED_FIELDS = [
    "name",
    "brand",
    "price",
    "description",
    "key_features",
    "image_urls",
    "category",
    "colors",
    "variants",
]
OPTIONAL_FIELDS = ["video_url"]


def diagnose_file(filepath: Path) -> dict:
    html = filepath.read_text(encoding="utf-8")
    parsed = parse_html(html)

    # Report parser-level extraction
    parser_stats = {
        "json_ld_blocks": len(parsed.json_ld),
        "og_tags": len(parsed.og_tags),
        "embedded_json_sources": list(parsed.embedded_json.keys()),
        "body_text_chars": len(parsed.body_text),
        "image_urls_from_html": len(parsed.image_urls),
        "video_urls_from_html": len(parsed.video_urls),
        "meta_tags": list(parsed.meta_tags.keys()),
    }

    # Run hydration only
    fields = _hydrate_fields(parsed)

    # Build report
    report = {"file": filepath.name, "parser": parser_stats, "fields": {}, "missing": [], "filled": []}

    for field in REQUIRED_FIELDS + OPTIONAL_FIELDS:
        val = fields.get(field)
        if val is None or val == "" or val == []:
            report["missing"].append(field)
            report["fields"][field] = None
        else:
            report["filled"].append(field)
            # Summarize the value
            if isinstance(val, Price):
                report["fields"][field] = f"{val.price} {val.currency}" + (
                    f" (was {val.compare_at_price})" if val.compare_at_price else ""
                )
            elif isinstance(val, list) and val and isinstance(val[0], Variant):
                report["fields"][field] = f"{len(val)} variants"
                # Show first 3
                for v in val[:3]:
                    report["fields"][field] += f"\n      {v.attributes} sku={v.sku} available={v.available}"
                if len(val) > 3:
                    report["fields"][field] += f"\n      ... and {len(val) - 3} more"
            elif isinstance(val, list):
                if all(isinstance(v, str) for v in val):
                    if len(val) <= 5:
                        report["fields"][field] = val
                    else:
                        report["fields"][field] = f"{len(val)} items: {val[:3]} + {len(val) - 3} more"
                else:
                    report["fields"][field] = f"{len(val)} items"
            elif isinstance(val, str):
                report["fields"][field] = val[:150] + ("..." if len(val) > 150 else "")
            else:
                report["fields"][field] = str(val)[:150]

    return report


def main():
    html_files = sorted(DATA_DIR.glob("*.html"))
    print(f"Diagnosing {len(html_files)} files (parser + hydration only, NO LLM)\n")

    all_reports = []
    for filepath in html_files:
        report = diagnose_file(filepath)
        all_reports.append(report)

        print(f"{'=' * 70}")
        print(f"  {report['file']}")
        print(f"{'=' * 70}")

        # Parser stats
        p = report["parser"]
        print(
            f"  Parser: {p['json_ld_blocks']} JSON-LD | {p['og_tags']} OG tags | "
            f"{len(p['embedded_json_sources'])} embedded JSON | "
            f"{p['image_urls_from_html']} imgs | {p['video_urls_from_html']} videos"
        )
        if p["embedded_json_sources"]:
            print(f"  Embedded sources: {p['embedded_json_sources']}")

        # Fields
        print(f"\n  Filled ({len(report['filled'])}/{len(REQUIRED_FIELDS)}):")
        for field in report["filled"]:
            val = report["fields"][field]
            if isinstance(val, list):
                print(f"    {field}: {val}")
            else:
                lines = str(val).split("\n")
                print(f"    {field}: {lines[0]}")
                for line in lines[1:]:
                    print(f"    {line}")

        if report["missing"]:
            print(f"\n  MISSING ({len(report['missing'])}): {report['missing']}")
        else:
            print("\n  All fields filled!")

        print()

    # Summary table
    print(f"\n{'=' * 70}")
    print("SUMMARY: Field coverage across all files")
    print(f"{'=' * 70}")
    print(f"{'Field':<20} ", end="")
    for r in all_reports:
        print(f"{r['file'][:12]:<14}", end="")
    print()
    print("-" * 90)

    for field in REQUIRED_FIELDS + OPTIONAL_FIELDS:
        print(f"{field:<20} ", end="")
        for r in all_reports:
            if field in r["filled"]:
                print(f"{'OK':<14}", end="")
            else:
                print(f"{'MISSING':<14}", end="")
        print()


if __name__ == "__main__":
    main()
