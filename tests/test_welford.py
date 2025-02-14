import numpy as np
import pandas as pd
from pytest import approx

from passengersim.tracers.welford import Welford


def test_welford_floats():
    w = Welford()
    w.update(1.0)
    w.update(2.0)
    w.update(3.0)
    assert w.mean == approx(2.0)
    assert w.variance == approx(0.6666666666666666)
    assert w.std_dev == approx(0.816496580927726)
    assert w.sample_variance == approx(1.0)
    assert w.sample_std_dev == approx(1.0)
    assert w.n == 3


def test_welford_tuples():
    w = Welford()
    w.update([1.0, 1.0, 1.0])
    w.update([2, 2, 2])
    w.update([3, 3, 3])
    assert w.mean == approx([2.0, 2.0, 2.0])
    assert w.variance == approx([0.6666666666666666] * 3)
    assert w.std_dev == approx([0.816496580927726] * 3)
    assert w.sample_variance == approx([1.0] * 3)
    assert w.sample_std_dev == approx([1.0] * 3)
    assert w.n == 3


def test_welford_arrays():
    w = Welford()
    w.update(np.ones([3, 3]))
    w.update(np.ones([3, 3]) * 2)
    w.update(np.ones([3, 3]) * 3)
    np.testing.assert_array_almost_equal(w.mean, np.ones([3, 3]) * 2)
    np.testing.assert_array_almost_equal(w.variance, np.ones([3, 3]) * 2 / 3)
    np.testing.assert_array_almost_equal(w.std_dev, np.sqrt(np.ones([3, 3]) * 2 / 3))
    np.testing.assert_array_almost_equal(w.sample_variance, np.ones([3, 3]))
    np.testing.assert_array_almost_equal(w.sample_std_dev, np.sqrt(np.ones([3, 3])))
    assert w.n == 3


def df_const(c):
    return pd.DataFrame({"Aa": [c] * 3, "Bb": [c] * 3, "Cc": [c] * 3})


def test_welford_dataframes():
    w = Welford()
    df = pd.DataFrame(
        {"Aa": [1.0, 2.0, 3.0], "Bb": [1.0, 2.0, 3.0], "Cc": [1.0, 2.0, 3.0]}
    )
    w.update(df)
    w.update(df + 1)
    w.update(df + 2)
    pd.testing.assert_frame_equal(w.mean, df + 1)
    pd.testing.assert_frame_equal(w.variance, df_const(0.6666666666666666))
    pd.testing.assert_frame_equal(w.std_dev, df_const(0.816496580927726))
    pd.testing.assert_frame_equal(w.sample_variance, df_const(1.0))
    pd.testing.assert_frame_equal(w.sample_std_dev, df_const(1.0))
    assert w.n == 3
