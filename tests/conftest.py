import os
import pathlib
from datetime import datetime

import pytest

pytest_plugins = ("passengersim.utils.regression_testing",)


def pytest_runtest_setup(item):
    for marker in item.iter_markers(name="skip_until"):
        #
        #     Usage:  @pytest.mark.skip_until("2025-12-25", responsible="jpn")
        #
        #     This allows developers to mark tests as "not ready", giving a data in the
        #     future when they should be ready to run. This way, we can temporarily
        #     disable tests that are not yet implemented or are known to fail, until
        #     a specific date when they will come back and run again.  This way,
        #     we can skip for now but still be sure it is not forgotten forever.
        #     The `responsible` keywork argument is optional and can be used to specify
        #     a responsible person (e.g. "jpn"). If provided, this test will always
        #     skip unless the current user matches the responsible party.
        #
        trigger_date = marker.args[0]
        responsible = marker.kwargs.get("responsible", "").lower()
        current_user = os.getenv("USER", "").lower()
        if responsible and current_user != responsible:
            pytest.skip(f"Skipping, {current_user} is not responsible")
        if datetime.now() < datetime.strptime(trigger_date, "%Y-%m-%d"):
            pytest.skip(f"Skipping until {trigger_date}")


@pytest.fixture(autouse=True)
def file_cleanup(request):

    yield  # do something in a test that should not create any files

    files = pathlib.Path.cwd().glob("passengersim-output.*")
    for f in files:
        raise AssertionError(f"file not cleaned up: {f}")
    files = pathlib.Path.cwd().glob("*.pxsim")
    for f in files:
        raise AssertionError(f"file not cleaned up: {f}")


def pytest_xdist_auto_num_workers(config):
    """
    Hooks into pytest-xdist auto-detection to use total cores minus 2.
    """
    # Use os.cpu_count() for total logical CPUs
    cpu_count = os.cpu_count() or 1

    # Calculate workers, ensuring at least 1, leaving 2 for system
    num_workers = max(1, cpu_count - 2)

    print(f"\n[xdist] Total CPUs: {cpu_count}, Using Workers: {num_workers}")
    return num_workers
