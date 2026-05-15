import ast
import csv
import math
import re
from pathlib import Path
from typing import Any

from passengersim.utils.compression import smart_open


def _preprocess_incoming_items(d):
    """Preprocess incoming dict items.

    - Remove keys where the value is None or NaN (for floats) or "nan", "none", "null", "" (for strings).
    - Convert values that appear to be string-formatted dicts (begin and end with curly brackets) to actual dicts.
    """
    out = {}

    # some regex tools to check for strings that are integers, floats, dicts, or lists
    _is_ast_literal_candidate = re.compile(
        r"^[0-9]+$"  # integer
        r"|^[0-9]+\.[0-9]+$"  # or float
        r"|^\{.*}$"  # or dict
        r"|^\[.*]$"  # or list
    )

    for k, v in d.items():
        if v is None:
            continue
        if isinstance(v, float) and math.isnan(v):
            continue
        if isinstance(v, str):
            if v.lower() in {"nan", "none", "null", ""}:
                continue
            if _is_ast_literal_candidate.match(v):
                try:
                    v = ast.literal_eval(v)  # use literal_eval to parse the string as a dict
                except Exception:
                    pass  # if eval fails, keep the original string value
        out[k] = v
    return out


def csv_to_list_of_dicts(content):
    reader = csv.DictReader(content)
    out = []
    for row in reader:
        out.append(_preprocess_incoming_items(row))  # row is already a dict
    return out


def load_from_csv(content: Any):
    """If the content is a path to a file, load the content, and return it."""
    if isinstance(content, str | Path):
        _content = Path(content)
        if _content.is_file():
            with smart_open(_content, mode="rt") as f:
                return csv_to_list_of_dicts(f)
        else:
            raise FileNotFoundError(_content)
    return content
