from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

    from passengersim import Simulation


class GenericTracer(ABC):
    name: str = "generic_tracer"

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def fresh(self) -> GenericTracer:
        raise NotImplementedError()

    @abstractmethod
    def attach(self, sim: Simulation) -> None:
        raise NotImplementedError()

    @abstractmethod
    def finalize(self) -> pd.DataFrame:
        raise NotImplementedError()
