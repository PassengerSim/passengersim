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
        """Reset the internal counters on this tracer to a null state."""
        raise NotImplementedError()

    @abstractmethod
    def fresh(self) -> GenericTracer:
        """Return a fresh copy of this tracer, tracing the same targets but not sharing state."""
        raise NotImplementedError()

    @abstractmethod
    def attach(self, sim: Simulation) -> None:
        """Attach this tracer to a Simulation, which will be traced as it runs."""
        raise NotImplementedError()

    @abstractmethod
    def finalize(self) -> pd.DataFrame:
        """Convert the internal counters on this tracer to a dataframe."""
        raise NotImplementedError()
