"""Shared helpers for the KV-cache quantization CSV aggregators
(test-kv-cache-quantization-{perf,perp}-agg.py)."""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys


def safe_float(v):
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def render_table(rows, display_cols, transform=None):
    """Render rows as a fixed-width text table.

    display_cols is a list of (row_key, column_title, width) tuples;
    transform, if given, maps (key, value) -> display value.
    """
    if not rows:
        return ""

    lines = []
    header = "  ".join(f"{title:>{w}}" for _, title, w in display_cols)
    lines.append(header)
    lines.append("  ".join("-" * w for _, _, w in display_cols))

    for row in rows:
        vals = []
        for k, _, w in display_cols:
            v = row.get(k, "")
            if transform is not None:
                v = transform(k, v)
            vals.append(f"{v:>{w}}")
        lines.append("  ".join(vals))

    return "\n".join(lines)


def run_aggregation(description, title, output_header, aggregate, compute_ratios,
                    render, legend=None):
    """argparse/IO skeleton shared by the aggregator scripts: parse args,
    aggregate the inputs, write the CSV and TXT outputs, and log the table."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("inputs", nargs="+", help="Input CSV files")
    parser.add_argument("-o", "--output", required=True,
                        help="Output CSV file path")
    args = parser.parse_args()

    log = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    missing = [f for f in args.inputs if not os.path.isfile(f)]
    if missing:
        log.error("Error: files not found: %s", ", ".join(missing))
        sys.exit(1)

    groups = aggregate(args.inputs)
    results = compute_ratios(groups)

    out_csv = args.output
    out_txt = os.path.splitext(out_csv)[0] + ".txt"

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=output_header,
                                quoting=csv.QUOTE_NONNUMERIC)
        writer.writeheader()
        writer.writerows(results)

    table = render(results)
    with open(out_txt, "w") as f:
        f.write("=" * 80 + "\n")
        f.write(title + "\n")
        f.write("=" * 80 + "\n\n")
        f.write(table + "\n\n")
        f.write("=" * 80 + "\n")
        f.write(f" Aggregated from {len(args.inputs)} file(s):\n")
        for fpath in args.inputs:
            f.write(f"   - {os.path.abspath(fpath)}\n")
        f.write("=" * 80 + "\n")
        if legend is not None:
            f.write("\n")
            f.write(legend)

    log.info("CSV: %s (%s rows)", os.path.abspath(out_csv), len(results))
    log.info("TXT: %s", os.path.abspath(out_txt))
    log.info("")
    log.info(table)
    log.info("")
    log.info("Aggregated from %s file(s):", len(args.inputs))
    for fpath in args.inputs:
        log.info("  - %s", fpath)
    if legend is not None:
        log.info("")
        log.info(legend)
