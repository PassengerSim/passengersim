"""Run completion must also finish and retire its progress display."""

import queue
from concurrent.futures import Future
from io import StringIO
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from rich.console import Console
from rich.progress import Progress

import passengersim as pax
from passengersim.driver._multiproc import ThrottledUpdater
from passengersim.experiments import Experiments
from passengersim.mp_executor import JobExecutor


class RecordingProgress(Progress):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, console=Console(file=StringIO()), **kwargs)
        self.removed = []
        self.last_frame = set()

    def refresh(self):
        self.last_frame = {task.id for task in self.tasks if task.visible}
        super().refresh()

    def remove_task(self, task_id):
        task = self._tasks[task_id]
        self.removed.append((task.completed, task.total, task.finished))
        super().remove_task(task_id)


@pytest.fixture
def config():
    cfg = pax.Config.from_yaml(pax.demo_network("3MKT/DEMO"))
    cfg.simulation_controls.num_trials = 2
    cfg.simulation_controls.num_samples = 3
    cfg.simulation_controls.burn_samples = 1
    cfg.simulation_controls.show_progress_bar = False
    cfg.outputs._write_no_files()
    cfg.outputs.reports.clear()
    cfg.db = None
    return cfg


@pytest.fixture
def immediate_workers(monkeypatch):
    """Finish all workers before the monitor can consume any queue messages."""
    import passengersim.driver._multiproc as multiproc
    import passengersim.mp_executor as executor_module

    pools = []

    class ImmediateExecutor:
        mode = "success"

        def __init__(self, **kwargs):
            self.closed = False
            pools.append(self)

        def submit(self, func, *args):
            future = Future()
            if self.mode == "cancel":
                future.cancel()
                future.set_running_or_notify_cancel()
            elif self.mode == "failure":
                future.set_exception(RuntimeError("worker failed"))
            else:
                try:
                    future.set_result(func(*args))
                except BaseException as exc:
                    future.set_exception(exc)
            return future

        def shutdown(self, **kwargs):
            self.closed = True

    monkeypatch.setattr(executor_module.concurrent.futures, "ProcessPoolExecutor", ImmediateExecutor)
    monkeypatch.setattr(executor_module, "Progress", RecordingProgress)
    monkeypatch.setattr(
        multiproc.multiprocessing, "Manager", lambda: SimpleNamespace(Queue=queue.Queue, shutdown=lambda: None)
    )
    return ImmediateExecutor, pools


def test_updater_counts_samples_not_initial_callback():
    updates = queue.Queue()
    updater = ThrottledUpdater(updates, trial_id=0, throttle=float("inf"))
    updater(0, 3)
    updater(1, 3)
    updater(2, 3)
    updater(3, 3)
    updater.flush()
    assert updates.get_nowait() == (0, "update", 3)
    assert updates.empty()


@pytest.mark.parametrize("mode", ["success", "failure", "cancel"])
def test_finished_workers_retire_tasks_without_queue_updates(config, immediate_workers, mode):
    executor, _ = immediate_workers
    executor.mode = mode
    progress = RecordingProgress(auto_refresh=False)
    unrelated = progress.add_task("Caller-owned task", total=10)
    sim = pax.MultiSimulation(config)
    try:
        if mode == "success":
            sim.run(max_processes=1, rich_progress=progress).close()
            assert progress.removed == [(3, 3, True), (3, 3, True)]
        elif mode == "failure":
            with pytest.raises(RuntimeError, match="worker failed"):
                sim.run(max_processes=1, rich_progress=progress)
            assert all(not finished for _, _, finished in progress.removed)
        else:
            with pytest.warns(UserWarning, match="no summaries available"):
                sim.run(max_processes=1, rich_progress=progress).close()
            assert all(not finished for _, _, finished in progress.removed)
        assert len(progress.removed) == 2
        assert progress.task_ids == [unrelated]
        assert progress.last_frame == {unrelated}
    finally:
        sim.close()


@pytest.mark.parametrize("failure_stage", [None, "report", "dispatch", "worker"])
def test_experiments_stop_display_and_join_monitors(config, tmp_path, immediate_workers, monkeypatch, failure_stage):
    executor, pools = immediate_workers
    if failure_stage == "worker":
        executor.mode = "failure"
    monitors = []
    displays = []
    original_run = pax.MultiSimulation._run_asynchronously

    def capture_run(sim, *args, **kwargs):
        if failure_stage == "dispatch" and monitors:
            raise RuntimeError("dispatch failed")
        displays.append(kwargs["jobber"].rich_progress)
        future = original_run(sim, *args, **kwargs)
        monitor = future._monitor_thread
        monitor.join = Mock(wraps=monitor.join)
        monitors.append(monitor)
        return future

    monkeypatch.setattr(pax.MultiSimulation, "_run_asynchronously", capture_run)
    experiments = Experiments(config, output_dir=tmp_path)

    @experiments
    def first(cfg):
        return cfg

    @experiments
    def second(cfg):
        return cfg

    if failure_stage == "report":
        monkeypatch.setattr(experiments, "_write_report_after_run", Mock(side_effect=RuntimeError("report failed")))
    if failure_stage:
        with pytest.raises(RuntimeError, match=f"{failure_stage} failed"):
            experiments.run(use_existing=False, write_report=True)
    else:
        experiments.run(use_existing=False, write_report=False)
    assert len(monitors) == (1 if failure_stage == "dispatch" else 2)
    assert all(monitor.join.called and not monitor.is_alive() for monitor in monitors)
    assert all(not display.live.is_started and not display.tasks for display in displays)
    assert all(pool.closed for pool in pools)


def test_owned_notebook_widget_is_closed(immediate_workers, monkeypatch):
    jobber = JobExecutor(max_workers=1)
    live = jobber.rich_progress.live
    # Exercise Rich's Jupyter stop path without requiring a real notebook frontend.
    live.console.is_jupyter = True
    monkeypatch.setattr(live, "refresh", lambda: None)
    widget = SimpleNamespace(close=Mock())
    with jobber:
        live.ipy_widget = widget
    widget.close.assert_called_once()
    assert not live.is_started


def test_real_workers_finish_and_clear_progress(config):
    progress = RecordingProgress(auto_refresh=False)
    with pax.MultiSimulation(config) as sim:
        result = sim.run(max_processes=2, rich_progress=progress)
        assert progress.removed == [(3, 3, True), (3, 3, True)]
        assert not progress.tasks and not progress.last_frame
        result.close()
