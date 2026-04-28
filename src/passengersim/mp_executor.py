import concurrent.futures
import multiprocessing
from types import TracebackType
from typing import Self

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
)

from .utils.caffeine import keep_awake


class JobExecutor:
    def __init__(self, max_workers: int | None = None):

        # initialize ProcessPoolExecutor
        if max_workers is None:
            max_workers = multiprocessing.cpu_count()
        self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)

        # initialize rich Progress
        self.rich_progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            auto_refresh=False,
        )

        self._number_of_entries = 0
        self._caffeine = None

    def start(self) -> Self:
        if self._number_of_entries == 0:
            self.rich_progress.start()
            self._caffeine = keep_awake()
            self._caffeine.__enter__()
        self._number_of_entries += 1
        return self

    def stop(self, kill: bool = False) -> None:
        if kill:
            self.rich_progress.stop()
            if self._caffeine is not None:
                self._caffeine.__exit__(None, None, None)
                self._caffeine = None
            self._number_of_entries = 0
        if self._number_of_entries > 0:
            self._number_of_entries -= 1
            if self._number_of_entries == 0:
                self._caffeine.__exit__(None, None, None)
                self._caffeine = None
                self.rich_progress.stop()

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self._number_of_entries -= 1
        if self._number_of_entries == 0:
            self._caffeine.__exit__(exc_type, exc_val, exc_tb)
            self._caffeine = None
            self.rich_progress.stop()

    def __del__(self) -> None:
        self.executor.shutdown(wait=False)
        if self._number_of_entries > 0:
            self.rich_progress.stop()
