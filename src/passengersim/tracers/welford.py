from __future__ import annotations

from typing import Any

import numpy as np


class Welford:
    def __init__(self):
        self._n = 0
        self._mean = 0
        self._mean2 = 0

    def update(self, x: np.typing.ArrayLike) -> None:
        self._n += 1
        if isinstance(x, tuple | list):
            x = np.array(x)
        delta = x - self._mean
        self._mean += delta / self._n
        self._mean2 += delta * (x - self._mean)

    @property
    def mean(self) -> np.typing.ArrayLike:
        return self._mean

    @property
    def variance(self) -> np.typing.ArrayLike:
        if self._n < 2:
            return np.nan
        return self._mean2 / self._n

    @property
    def std_dev(self) -> np.typing.ArrayLike:
        return np.sqrt(self.variance)

    @property
    def sample_variance(self) -> np.typing.ArrayLike:
        if self._n < 2:
            return np.nan
        return self._mean2 / (self._n - 1)

    @property
    def sample_std_dev(self) -> np.typing.ArrayLike:
        return np.sqrt(self.sample_variance)

    @property
    def n(self) -> int:
        return self._n


class MultiWelford:
    def __init__(self, keys: list[str], aux: dict[str, Any] | set[str] = None):
        """
        Initialize a MultiWelford object.

        Parameters
        ----------
        keys : list[str]
            The keys of the input dict.
        aux : dict[str, Any], optional
            Auxiliary data to store with the statistics.  This can be labels
            or other constant data, which is not tracked by the online statistics
            algorithm, but is useful for plotting or other analysis and is stored
            and returned with the statistics.
        """
        self._n = 0
        self._mean = {k: 0 for k in keys}
        self._mean2 = {k: 0 for k in keys}
        self._aux = aux or {}

    def update(self, x: dict[str, np.typing.ArrayLike]) -> None:
        self._n += 1
        if not isinstance(x, dict) and hasattr(x, "__dict__"):
            x = x.__dict__
        if isinstance(self._aux, set):
            # convert set of key to dict on first update
            self._aux = {k: x[k] for k in self._aux}
        for k in self._mean.keys():
            # missing keys are ignored, equivalent to treating them as zero
            if k in x:
                delta = x[k] - self._mean[k]
                self._mean[k] += delta / self._n
                self._mean2[k] += delta * (x[k] - self._mean[k])

    @property
    def mean(self) -> dict[str, np.typing.ArrayLike]:
        return self._aux | {k: v for (k, v) in self._mean.items()}

    @property
    def variance(self) -> dict[str, np.typing.ArrayLike]:
        if self._n < 2:
            return {k: np.nan for k in self._mean.keys()}
        return self._aux | {k: (self._mean2[k] / self._n) for k in self._mean.keys()}

    @property
    def std_dev(self) -> dict[str, np.typing.ArrayLike]:
        return self._aux | {k: np.sqrt(v) for k, v in self.variance.items()}

    @property
    def sample_variance(self) -> dict[str, np.typing.ArrayLike]:
        if self._n < 2:
            return {k: np.nan for k in self._mean.keys()}
        return self._aux | {k: (self._mean2[k] / (self._n - 1)) for k in self._mean.keys()}

    @property
    def sample_std_dev(self) -> dict[str, np.typing.ArrayLike]:
        return self._aux | {k: np.sqrt(v) for k, v in self.sample_variance.items()}

    @property
    def n(self) -> int:
        return self._n


class WeightedWelford:
    def __init__(self):
        self._w_sum = 0
        self._w_sum2 = 0
        self._mean = 0
        self._S = 0

    def update(self, x: np.typing.ArrayLike, w: np.typing.ArrayLike) -> None:
        self._w_sum += w
        self._w_sum2 += w**2
        mean_old = self.mean
        self._mean = mean_old + (w / self._w_sum) * (x - mean_old)
        self.S = self._S + w * (x - mean_old) * (x - self.mean)

    @property
    def mean(self) -> np.typing.ArrayLike:
        return self._mean

    @property
    def variance(self) -> np.typing.ArrayLike:
        return self._S / self._w_sum

    @property
    def std_dev(self) -> np.typing.ArrayLike:
        return np.sqrt(self.variance)

    @property
    def sample_variance(self) -> np.typing.ArrayLike:
        return self._S / (self._w_sum - 1)

    @property
    def sample_std_dev(self) -> np.typing.ArrayLike:
        return np.sqrt(self.sample_variance)

    @property
    def n(self) -> int:
        return self._w_sum
