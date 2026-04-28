from __future__ import annotations

from typing import Any

_NO_DEFAULT = object()


class RmSysOption:
    def __init__(
        self, name: str, default: Any = _NO_DEFAULT, doc: str = None, expected_type: Any = _NO_DEFAULT
    ) -> None:
        self.name = name
        self.default = default
        self.doc = doc
        self.expected_type = expected_type

    def has_default(self) -> bool:
        return self.default is not _NO_DEFAULT
