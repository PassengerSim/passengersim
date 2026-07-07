from __future__ import annotations

import pathlib
from typing import Literal


class SnapshotInstruction:
    def __init__(
        self,
        trigger: bool = False,
        filepath: pathlib.Path | str | None = None,
        *,
        why: str | None = None,
        mode: Literal["w", "a"] = "w",
    ):
        self.trigger = bool(trigger)
        """Has this snapshot been triggered."""

        self.why = why
        """Explanation of why snapshot is (or is not) triggered."""

        if isinstance(filepath, str):
            filepath = pathlib.Path(filepath)

        self.filepath = filepath
        """Where to save snapshot content."""

        self.mode = mode
        """Write mode for new content, `w` overwrites existing file, `a` appends."""

    def __bool__(self) -> bool:
        return self.trigger

    def write(self, content: str = ""):
        """Write snapshot content to a file, or just print it"""
        if not content:
            return
        if self.filepath:
            with self.filepath.open(mode=self.mode) as f:
                if self.why:
                    f.write(self.why)
                    f.write("\n")
                if isinstance(content, bytes):
                    content = content.decode("utf-8")
                elif not isinstance(content, str):
                    content = str(content)
                f.write(content)
                if content[-1] != "\n":
                    f.write("\n")
        else:
            if self.why:
                print(self.why)
            print(content)

    def write_more(self, content: str = ""):
        """Write additional snapshot content to a file, or just print it"""
        if not content:
            return
        if self.filepath:
            with self.filepath.open(mode="a") as f:
                if isinstance(content, bytes):
                    content = content.decode("utf-8")
                elif not isinstance(content, str):
                    content = str(content)
                f.write(content)
                if content[-1] != "\n":
                    f.write("\n")
        else:
            print(content)
