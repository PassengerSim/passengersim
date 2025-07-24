from __future__ import annotations

import pathlib
import time
from typing import Literal

from pydantic import BaseModel, field_validator


class SnapshotInstruction:
    def __init__(
        self,
        trigger: bool = False,
        filepath: pathlib.Path | None = None,
        why: str | None = None,
        filter: SnapshotFilter | None = None,
        mode: Literal["w", "a"] = "w",
    ):
        self.trigger = bool(trigger)
        """Has this snapshot been triggered."""
        self.why = why
        """Explanation of why snapshot is (or is not) triggered."""
        self.filepath = filepath
        """Where to save snapshot content."""
        self.filter = filter
        """A reference to the filter that spawned this instruction."""
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


class SnapshotFilter(BaseModel, validate_assignment=True):
    type: Literal[
        "fare_adj",
        "forecast",
        "leg_untruncation",
        "path_untruncation",
        "rm",
        "pro_bp",
        "forecast_adj",
        "hybrid",
        "udp",
        None,
    ] = None
    title: str = ""
    carrier: str = ""
    trial: list[int] = []
    sample: list[int] = []
    dcp: list[int] = []
    orig: list[str] = []
    dest: list[str] = []
    flt_no: list[int] = []
    logger: str | None = None
    directory: pathlib.Path | None = None

    @field_validator("trial", "sample", "dcp", "orig", "dest", "flt_no", mode="before")
    def _allow_singletons(cls, v):
        """Allow a singleton value that is converted to a list of one item."""
        if not isinstance(v, list | tuple):
            v = [v]
        return v

    def filepath(self, sim, leg=None, path=None) -> pathlib.Path | None:
        if self.directory is None:
            return None
        pth = self.directory
        if leg is not None:
            pth = pth.joinpath(f"carrier-{leg.carrier}")
        pth = pth.joinpath(f"dpc-{sim.last_dcp}")
        if leg is not None:
            pth = pth.joinpath(f"orig-{leg.orig}")
        elif path is not None:
            pth = pth.joinpath(f"orig-{path.orig}")
        if leg is not None:
            pth = pth.joinpath(f"dest-{leg.dest}")
        elif path is not None:
            pth = pth.joinpath(f"dest-{path.dest}")
        if leg is not None:
            pth = pth.joinpath(f"fltno-{leg.flt_no}")
        elif path is not None:
            pth = pth.joinpath(f"fltno-{path.get_leg_fltno(0)}")
        if sim.num_trials > 1:
            pth = pth.joinpath(f"trial-{sim.trial}")
        pth = pth.joinpath(f"sample-{sim.sample}")
        pth.parent.mkdir(parents=True, exist_ok=True)
        return pth.with_suffix(".log")

    def run(self, sim, leg=None, path=None, carrier=None, orig=None, dest=None, why=False) -> SnapshotInstruction:
        # Check the filter conditions
        info = ""

        if len(self.trial) > 0 and sim.trial not in self.trial and sim.num_trials > 1:
            return SnapshotInstruction(False, why=f"cause {sim.trial=}")
        info += f"  trial={sim.trial}"

        if len(self.sample) > 0 and sim.sample not in self.sample:
            return SnapshotInstruction(False, why=f"cause {sim.sample=}")
        info += f"  sample={sim.sample}"

        if len(self.dcp) > 0 and sim.last_dcp not in self.dcp:
            return SnapshotInstruction(False, why=f"cause {sim.last_dcp=}")
        info += f"  dcp={sim.last_dcp}"

        if leg is not None:
            if self.carrier and leg.carrier != self.carrier:
                return SnapshotInstruction(False, why=f"cause {leg.carrier=}")
            info += f"  carrier={leg.carrier}"

            if len(self.orig) > 0 and leg.orig not in self.orig:
                return SnapshotInstruction(False, why=f"cause {leg.orig=}")
            info += f"  orig={leg.orig}"

            if len(self.dest) > 0 and leg.dest not in self.dest:
                return SnapshotInstruction(False, why=f"cause {leg.dest=}")
            info += f"  dest={leg.dest}"

            if len(self.flt_no) > 0 and leg.flt_no not in self.flt_no:
                return SnapshotInstruction(False, why=f"cause {leg.flt_no=}")
            info += f"  flt_no={leg.flt_no}"

        if path is not None:
            if len(self.orig) > 0 and path.orig not in self.orig:
                return SnapshotInstruction(False, why=f"cause {path.orig=}")
            info += f"  orig={path.orig}"

            if len(self.dest) > 0 and path.dest not in self.dest:
                return SnapshotInstruction(False, why=f"cause {path.dest=}")
            info += f"  dest={path.dest}"

            if len(self.flt_no) > 0 and path.get_leg_fltno(0) not in self.flt_no:
                return SnapshotInstruction(False, why=f"cause {path.get_leg_fltno(0)=}")
            info += f"  flt_no={path.get_leg_fltno(0)}"

        if carrier is not None:
            if self.carrier and carrier != self.carrier:
                return SnapshotInstruction(False, why=f"cause {carrier=}")
            info += f"  carrier={carrier}"

        if orig is not None:
            if self.orig and orig not in self.orig:
                return SnapshotInstruction(False, why=f"cause {orig=}")
            info += f"  orig={orig}"

        if dest is not None:
            if self.carrier and dest not in self.dest:
                return SnapshotInstruction(False, why=f"cause {dest=}")
            info += f"  dest={dest}"

        # Now do something
        snapshot_file = self.filepath(sim, leg, path)
        created_date = time.strftime("Snapshot created %Y-%m-%d %A %I:%M:%S %p")
        title = f"{self.title}:{info}\n{created_date}\n"
        if len(self.title) > 0 and not snapshot_file:
            print(f"{self.title}:{info}", flush=True)

        self._last_run_info = info

        if self.type in ["fare_adj"]:
            return SnapshotInstruction(True, snapshot_file, why=title, filter=self)
        elif self.type in ["leg_untruncation", "path_untruncation"]:
            return SnapshotInstruction(True, snapshot_file, why=title, filter=self)
        elif self.type in ("forecast", "forecast_adj"):
            return SnapshotInstruction(True, snapshot_file, why=title, filter=self)
        elif self.type == "hybrid":
            return SnapshotInstruction(True, snapshot_file, why=title, filter=self)
        elif self.type == "rm":
            bucket_detail = leg.print_bucket_detail()
            snapshot_file = self.filepath(sim, leg, path)
            if snapshot_file:
                with snapshot_file.open(mode="a") as f:
                    f.write(title)
                    f.write(bucket_detail)
            else:
                print(bucket_detail)
            return SnapshotInstruction(True, snapshot_file, why=title, filter=self)
        elif self.type == "pro_bp":
            return SnapshotInstruction(True, snapshot_file, why=title, filter=self)
        elif self.type == "udp":
            return SnapshotInstruction(True, snapshot_file, why=title, filter=self)

        raise ValueError("unknown snapshot filter type")
        # return SnapshotInstruction(False, why="cause unknown")
