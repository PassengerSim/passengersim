from typing import Any

from pydantic import BaseModel


class PrettyModel(BaseModel):
    """Pretty-print as YAML style outputs."""

    def __repr_with_indent__(self, indent=0):
        x = []
        i = " " * indent
        for k, v in self:
            try:
                val = v.__repr_with_indent__(0)
            except AttributeError:
                if isinstance(v, dict):
                    val = repr_dict_with_indent(v, indent)
                else:
                    val = repr(v)
                    if isinstance(v, list) and len(val) > 70:
                        val = "- " + "\n- ".join(repr(j) for j in v)
            if "\n" in val:
                val_lines = val.split("\n")
                val = "\n  " + "\n  ".join(val_lines)
            elif isinstance(v, list):
                val = "\n  - " + "\n  - ".join(repr(j) for j in v)
            x.append(f"{i}{k}: {val}")
        return "\n".join(x)

    def __repr__(self):
        return f"{self.__class__.__name__}:\n" + self.__repr_with_indent__(2)


def repr_dict_with_indent(d: dict[str, Any], indent=0):
    x = []
    i = " " * indent
    for k, v in d.items():
        try:
            val = v.__repr_with_indent__(indent)
        except AttributeError:
            if isinstance(v, dict):
                val = repr_dict_with_indent(v, indent)
            else:
                val = repr(v)
                if isinstance(v, list) and len(val) > 70:
                    val = "- " + "\n- ".join(repr(j) for j in v)
        if "\n" in val:
            val_lines = val.split("\n")
            val = "\n  " + "\n  ".join(val_lines)
        elif isinstance(v, list):
            val = "\n  - " + "\n  - ".join(repr(j) for j in v)
        x.append(f"{i}{k}: {val}")
    return "\n".join(x)
