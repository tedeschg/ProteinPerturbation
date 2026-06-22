#!/usr/bin/env python3
"""
parse_config.py — robust YAML reader for run_pipeline.sh.

Reads a YAML file and prints the value of a dotted key path to stdout.
Used by the bash pipeline as a drop-in replacement for the previous
grep/awk-based YAML parser, which was brittle around nested keys,
comments, and quoting.

Usage:
    python parse_config.py <yaml_file> <dotted_key>

Example:
    python parse_config.py config.yaml residues_selection.output_file
    python parse_config.py config.yaml ligandmpnn.temperature

Exit codes:
    0 — value printed
    1 — file missing or key not found
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.stderr.write(
        "ERROR: PyYAML not found. Install with `pip install pyyaml` "
        "or `conda install pyyaml`.\n"
    )
    sys.exit(1)


def lookup(data, dotted_key: str):
    """Walk a dotted key path through a nested dict; return None if any
    segment is missing."""
    cur = data
    for segment in dotted_key.split("."):
        if not isinstance(cur, dict) or segment not in cur:
            return None
        cur = cur[segment]
    return cur


def format_value(value) -> str:
    """Render a YAML value as a single-line shell-friendly string."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return " ".join(str(x) for x in value)
    return str(value)


def main() -> int:
    if len(sys.argv) != 3:
        sys.stderr.write(
            "Usage: parse_config.py <yaml_file> <dotted_key>\n"
        )
        return 1

    yaml_path = Path(sys.argv[1])
    dotted    = sys.argv[2]

    if not yaml_path.is_file():
        sys.stderr.write(f"ERROR: YAML file not found: {yaml_path}\n")
        return 1

    with yaml_path.open() as f:
        data = yaml.safe_load(f)

    value = lookup(data, dotted)
    if value is None:
        sys.stderr.write(f"ERROR: key not found: {dotted}\n")
        return 1

    # Expand ${VAR} references against the current environment
    rendered = os.path.expandvars(format_value(value))
    print(rendered)
    return 0


if __name__ == "__main__":
    sys.exit(main())
