from __future__ import annotations

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
