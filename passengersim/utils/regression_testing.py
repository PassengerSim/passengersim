from __future__ import annotations

import io
import os
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from pytest_regressions.common import perform_regression_check

from .compression import deserialize_from_file, serialize_to_file


def nested_comparison(left, right, pd_opts=None) -> dict:
    differences = {}
    if pd_opts is None:
        pd_opts = {}
    if isinstance(left, dict) and isinstance(right, dict):
        left_missing_keys = set(left.keys()) - set(right.keys())
        right_missing_keys = set(right.keys()) - set(left.keys())
        if left_missing_keys:
            differences["left_missing_keys"] = left_missing_keys
        if right_missing_keys:
            differences["right_missing_keys"] = right_missing_keys
        for k in left.keys() & right.keys():
            diff = nested_comparison(left[k], right[k])
            if diff:
                differences[k] = diff
    elif isinstance(left, list | tuple) and isinstance(right, list | tuple):
        if len(left) != len(right):
            differences["length"] = len(left), len(right)
        for i, (lefty, righty) in enumerate(zip(left, right)):
            diff = nested_comparison(lefty, righty)
            if diff:
                differences[i] = diff
    elif isinstance(left, pd.DataFrame) and isinstance(right, pd.DataFrame):
        try:
            pd.testing.assert_frame_equal(left, right, **pd_opts)
        except AssertionError as e:
            differences["dataframe"] = str(e) or e
    elif isinstance(left, pd.Series) and isinstance(right, pd.Series):
        try:
            pd.testing.assert_series_equal(left, right, **pd_opts)
        except AssertionError as e:
            differences["series"] = str(e) or e
    elif isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        try:
            np.testing.assert_array_almost_equal(left, right)
        except AssertionError as e:
            differences["array"] = str(e) or e
    elif isinstance(left, str) and isinstance(right, str):
        if left != right:
            differences["string"] = f"{left!r} != {right!r}"
    elif hasattr(left, "__dict__") and hasattr(right, "__dict__"):
        diff = nested_comparison(left.__dict__, right.__dict__)
        if diff:
            differences["__dict__"] = diff
    else:
        try:
            assert left == pytest.approx(right)
        except AssertionError as e:
            if not str(e):
                msg = f"{left!r} != {right!r}"
            else:
                msg = str(e)
            differences["value"] = msg or e
    return differences


def print_diff(diff, prefix="", file=None):
    for k, v in diff.items():
        if isinstance(v, dict):
            if prefix:
                print_diff(v, prefix=f"{prefix}.{k}", file=file)
            else:
                print_diff(v, prefix=str(k), file=file)
        else:
            print(f"\033[91m{prefix}:\033[0m", file=file)
            if str(v):
                print(textwrap.indent(str(v), "  "), file=file)
            else:
                print(f"  does not match {type(v)}", file=file)


def deep_compare_obj(a_: Any, b_: Any):
    diff = nested_comparison(a_, b_)
    if diff:
        buffer = io.StringIO()
        print_diff(diff, file=buffer)
        raise AssertionError(f"deep comparison failure:\n{buffer.getvalue()}")


def deep_compare(a: os.PathLike, b: os.PathLike):
    a_ = deserialize_from_file(str(a))
    b_ = deserialize_from_file(str(b))
    deep_compare_obj(a_, b_)


class DeepRegressionFixture:
    """
    Implementation of `deepdiff_regression` fixture.
    """

    def __init__(
        self, datadir: Path, original_datadir: Path, request: pytest.FixtureRequest
    ) -> None:
        self.request = request
        self.datadir = datadir
        self.original_datadir = original_datadir
        self.force_regen = False
        self.with_test_class_names = False

    def check(
        self,
        data: Any,
        basename: str | None = None,
        fullpath: os.PathLike[str] | None = None,
    ) -> None:
        """
        Checks the data against a previously recorded version, or generate a new file.

        Parameters
        ----------
        data : Any
            any serializable data.

        basename : Optional[str], optional
            Basename of the file to test/record. If not given the name of the test is
            used. Use either `basename` or `fullpath`.

        fullpath : Path-like, optional
            complete path to use as a reference file. This option will ignore `datadir`
            fixture when reading *expected* files but will still use it to write
            *obtained* files. Useful if a reference file is located in the session
            data dir for example.
        """
        __tracebackhide__ = True

        # if round_digits is not None:
        #     round_digits_in_data(data_dict, round_digits)

        def dump(filename: Path) -> None:
            """Dump dict contents to the given filename"""
            serialize_to_file(str(filename), data)

        perform_regression_check(
            datadir=self.datadir,
            original_datadir=self.original_datadir,
            request=self.request,
            check_fn=deep_compare,
            dump_fn=dump,
            extension=".pkl.lz4",
            basename=basename,
            fullpath=fullpath,
            force_regen=self.force_regen,
            with_test_class_names=self.with_test_class_names,
        )


@pytest.fixture
def deep_regression(
    datadir: Path, original_datadir: Path, request: pytest.FixtureRequest
) -> DeepRegressionFixture:
    """
    Fixture used to test arbitrary data against known versions previously
    recorded by this same fixture.

    The data may be anything serializable by the `serialize_to_file` function in
    the `passengersim.utils.compression` module.
    """

    return DeepRegressionFixture(datadir, original_datadir, request)


def deep_regression_check(data: Any, filename: Path):
    """Check data against the given filename."""
    deep_compare_obj(data, deserialize_from_file(str(filename)))
