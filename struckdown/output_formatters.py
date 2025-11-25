import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Union

from jinja2 import Environment, FileSystemLoader, Template, Undefined

logger = logging.getLogger(__name__)


def flatten_dict(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten nested dictionaries for CSV/Excel output."""
    items = {}
    for k, v in d.items():
        new_key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key))
        elif isinstance(v, list):
            # Convert lists to comma-separated strings
            items[new_key] = ", ".join(str(item) for item in v)
        else:
            items[new_key] = v
    return items


def write_json(data: List[Dict[str, Any]], output_path: Path) -> None:
    """Write data to JSON file."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"Wrote {len(data)} results to {output_path}")


def write_csv(data: List[Dict[str, Any]], output_path: Path) -> None:
    """Write data to CSV file."""
    if not data:
        logger.warning("No data to write to CSV")
        return

    # Flatten all records and collect all unique keys
    flattened = [flatten_dict(record) for record in data]
    all_keys = set()
    for record in flattened:
        all_keys.update(record.keys())

    # check if we have original column ordering from spreadsheet input
    original_columns = None
    for record in data:
        if "_original_columns" in record:
            original_columns = record["_original_columns"]
            break

    if original_columns:
        # preserve original column order, append new columns in appearance order
        # exclude _original_columns itself from output
        all_keys.discard("_original_columns")
        # only include original columns that actually exist in output
        existing_orig_cols = [col for col in original_columns if col in all_keys]
        # preserve order by using first record's keys (dicts maintain insertion order in Python 3.7+)
        new_columns = [k for k in flattened[0].keys() if k not in original_columns and k != "_original_columns"]
        fieldnames = existing_orig_cols + new_columns
    else:
        # default: alphabetical order
        fieldnames = sorted(all_keys)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flattened)

    logger.info(f"Wrote {len(data)} results to {output_path}")


def write_xlsx(data: List[Dict[str, Any]], output_path: Path) -> None:
    """Write data to Excel file."""
    try:
        import openpyxl
        from openpyxl import Workbook
    except ImportError:
        raise ImportError(
            "openpyxl is required for Excel output. Install with: pip install openpyxl"
        )

    if not data:
        logger.warning("No data to write to Excel")
        return

    # Flatten all records and collect all unique keys
    flattened = [flatten_dict(record) for record in data]
    all_keys = set()
    for record in flattened:
        all_keys.update(record.keys())

    # check if we have original column ordering from spreadsheet input
    original_columns = None
    for record in data:
        if "_original_columns" in record:
            original_columns = record["_original_columns"]
            break

    if original_columns:
        # preserve original column order, append new columns in appearance order
        # exclude _original_columns itself from output
        all_keys.discard("_original_columns")
        # only include original columns that actually exist in output
        existing_orig_cols = [col for col in original_columns if col in all_keys]
        # preserve order by using first record's keys (dicts maintain insertion order in Python 3.7+)
        new_columns = [k for k in flattened[0].keys() if k not in original_columns and k != "_original_columns"]
        fieldnames = existing_orig_cols + new_columns
    else:
        # default: alphabetical order
        fieldnames = sorted(all_keys)

    wb = Workbook()
    ws = wb.active
    ws.title = "Results"

    # Write header
    ws.append(fieldnames)

    # Write data
    for record in flattened:
        row = [record.get(key, "") for key in fieldnames]
        ws.append(row)

    wb.save(output_path)
    logger.info(f"Wrote {len(data)} results to {output_path}")


def write_markdown(data: List[Dict[str, Any]], output_path: Path) -> None:
    """Write data to Markdown table."""
    if not data:
        logger.warning("No data to write to Markdown")
        return

    # Flatten all records and collect all unique keys
    flattened = [flatten_dict(record) for record in data]
    all_keys = set()
    for record in flattened:
        all_keys.update(record.keys())

    fieldnames = sorted(all_keys)

    lines = []

    # Header
    lines.append("| " + " | ".join(fieldnames) + " |")
    lines.append("| " + " | ".join(["---"] * len(fieldnames)) + " |")

    # Data rows
    for record in flattened:
        values = [str(record.get(key, "")) for key in fieldnames]
        lines.append("| " + " | ".join(values) + " |")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info(f"Wrote {len(data)} results to {output_path}")


def render_template(
    data: List[Dict[str, Any]], output_path: Path, template_path: Path
) -> None:
    """
    Render data through Jinja2 template.

    The template receives the entire results list as 'results' variable.
    Templates can iterate over results: {% for r in results %}...{% endfor %}
    """
    if not template_path.exists():
        raise FileNotFoundError(f"Template file not found: {template_path}")

    # Load template with tolerant undefined handling (missing vars become empty)
    env = Environment(
        loader=FileSystemLoader(template_path.parent), undefined=Undefined
    )
    template = env.get_template(template_path.name)

    # Render template once with all results
    final_output = template.render(results=data)

    # Write output
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_output)

    logger.info(
        f"Rendered {len(data)} results using template {template_path.name} to {output_path}"
    )


def write_output(
    data: List[Dict[str, Any]], output_path: Union[Path, str, None] = None
) -> None:
    """
    Write data to output file with format auto-detection.
    If output_path is None, writes JSON to stdout.
    """
    if output_path is None:
        # Write to stdout as JSON
        print(json.dumps(data, indent=2, ensure_ascii=False, default=str))
        return

    output_path = Path(output_path)
    extension = output_path.suffix.lower()

    if extension == ".json":
        write_json(data, output_path)
    elif extension == ".csv":
        write_csv(data, output_path)
    elif extension in [".xlsx", ".xls"]:
        write_xlsx(data, output_path)
    elif extension in [".md", ".txt"]:
        write_markdown(data, output_path)
    else:
        raise ValueError(
            f"Unsupported output format: {extension}. "
            "Supported formats: .json, .csv, .xlsx, .md, .txt"
        )
