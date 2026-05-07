from __future__ import annotations

import concurrent.futures
import json
import multiprocessing
import pathlib
import threading
import time
import traceback
import typing
import uuid
import warnings
from datetime import UTC, datetime

import dill
import numpy as np  # noqa: F401
import pandas as pd  # noqa: F401
from rich.traceback import Traceback as _RichTraceback

from .callbacks import CallbackMixin
from .config import Config
from .core import SimulationEngine
from .driver import BaseSimulation, Simulation, check_summarizer
from .mp_executor import JobExecutor
from .rm.systems import export_registered_rm_systems, restore_registered_rm_systems
from .summaries import GenericSimulationTables, SimulationTables
from .utils.tempdir import MaybeTemporaryDirectory

if typing.TYPE_CHECKING:
    from rich.progress import Progress


class ThrottledUpdater:
    def __init__(self, progress_queue, trial_id, throttle: float = 0.5):
        self.progress_queue = progress_queue
        self.trial_id = trial_id
        self.throttle = throttle
        self.last_time = 0
        self.n = 0

    def __call__(self, *args, **kwargs):
        self.n += 1
        if time.time() - self.last_time > self.throttle:
            self.flush()

    def flush(self):
        self.progress_queue.put((self.trial_id, "update", self.n))
        self.last_time = time.time()
        self.n = 0


def _subprocess_run_trial(
    trial_id: int,
    cfg_json: str,
    progress_queue: multiprocessing.Queue,
    output_dir: pathlib.Path | None = None,
    *,
    summarizer=SimulationTables,
    callbacks=None,
    exported_rm_systems=None,
):
    try:
        if exported_rm_systems:
            restore_registered_rm_systems(exported_rm_systems)
        cfg = Config.model_validate(json.loads(cfg_json))
        # Never write basic outputs to disk from a subprocess, except possibly for sim_state
        cfg.outputs.disk = False
        cfg.outputs.html.filename = False
        cfg.outputs.pickle = False
        cfg.outputs.excel = False
        # Writing to database is not a "output" and so is still allowed
        # but we need to make sure that each subprocess has a unique database file
        # because SQLite does not like multiple processes writing to the same database file.
        if cfg.db is not None and cfg.db.filename is not None and str(cfg.db.filename) != ":memory:":
            cfg.db.filename = cfg.db.filename.with_suffix(f".trial{trial_id:02}" + cfg.db.filename.suffix)
        output_dir = MaybeTemporaryDirectory(output_dir)

        sim = Simulation(cfg, output_dir.joinpath(f"passengersim-trial-{trial_id:02}"))
        if callbacks is not None:
            for cb in callbacks.get("begin_sample_callbacks", []):
                sim.begin_sample_callback(cb)
            for cb in callbacks.get("end_sample_callbacks", []):
                sim.end_sample_callback(cb)
            for cb in callbacks.get("daily_callbacks", []):
                sim.daily_callback(cb)
        sim.sample_done_callback = ThrottledUpdater(progress_queue, trial_id, 0.5)
        summary = sim.run(single_trial=trial_id, summarizer=summarizer)
        sim.sample_done_callback.flush()
        progress_queue.put((trial_id, "finalizing", None))
        # Passing a database connection between processes is not allowed,
        # so we need to delete it before returning the summary. But first
        # we will run any queries that were requested as output reports
        if isinstance(summary, GenericSimulationTables):
            summary.run_queries(items=cfg.outputs.reports)
        try:
            summary.cnx = None
        except AttributeError as err:
            print(err)
        progress_queue.put((trial_id, "done", summary))
        return summary
    except Exception as e:
        # Capture the subprocess's full traceback text so it can be preserved
        # when the exception is pickled and re-raised in the parent process.
        # Exception traceback objects (``__traceback__``) are not picklable, so
        # we attach the formatted traceback string via ``add_note`` (PEP 678).
        # Notes survive pickling and are automatically displayed by Python's
        # default traceback printer when the exception is eventually raised.
        tb_str = traceback.format_exc()
        note = f"Subprocess traceback (trial {trial_id}):\n{tb_str.rstrip()}"
        try:
            e.add_note(note)
        except AttributeError:
            # Fallback for very old Python versions lacking ``add_note``.
            existing = getattr(e, "__notes__", [])
            e.__notes__ = [*existing, note]
        # Also store the formatted traceback as a plain attribute so the
        # parent process can display it directly (e.g., in the rich console)
        # without needing to re-raise the exception.
        try:
            e._passengersim_subprocess_tb = tb_str
        except Exception:
            pass
        progress_queue.put((trial_id, "exception", e))
        return e


def _executor_run_trial(
    trial_id: int,
    cfg_json: str,
    progress_queue,
    output_dir: pathlib.Path | None,
    dill_kwargs_blob: bytes,
):
    """Worker function for use with ProcessPoolExecutor.

    Deserializes dill-encoded kwargs and delegates to _subprocess_run_trial.
    Exceptions are re-raised so the Future captures them.
    """
    kwargs = dill.loads(dill_kwargs_blob)
    result = _subprocess_run_trial(trial_id, cfg_json, progress_queue, output_dir, **kwargs)
    if isinstance(result, Exception):
        raise result
    return result


class MultiSimulation(BaseSimulation, CallbackMixin):
    def __init__(
        self,
        config: Config,
        output_dir: pathlib.Path | None = None,
    ):
        super().__init__(config, output_dir)
        self.config = config
        self._placeholder_sim = None

    def sequential_run(self, *, summarizer=SimulationTables):
        results = []
        cfg_json = self.config.model_dump_json()
        for trial_id in range(self.config.simulation_controls.num_trials):
            print("starting trial", trial_id)
            results.append(
                _subprocess_run_trial(
                    trial_id,
                    cfg_json,
                    output_dir=self.output_dir,
                    summarizer=summarizer,
                    callbacks=self.callback_functions(),
                )
            )
            print("finished trial", trial_id)
        return SimulationTables.aggregate(results)

    @property
    def _eng(self) -> SimulationEngine:
        if self._placeholder_sim is None:
            self._placeholder_sim = Simulation(self.config, self.output_dir)
        return self._placeholder_sim

    def _dump_config(self) -> str:
        """
        Dump the configuration to a JSON string suitable for a subprocess.

        License info is added to the config if available.  Output file
        reporting is removed.

        Returns
        -------
        str
        """
        if self.config.raw_license_certificate is None:
            try:
                from passengersim_license import raw_license_certificate
            except ImportError:
                raw_license_certificate = None
            self.config.raw_license_certificate = raw_license_certificate
        return self.config.model_dump_json(exclude={"outputs": {"html", "pickle", "excel", "log_reports", "disk"}})

    def _run_asynchronously(
        self,
        *,
        summarizer: type | None = None,
        output_dir: pathlib.Path | None = None,
        fail_fast: bool = True,
        run_id: str | None = None,
        jobber: JobExecutor,
        rich_progress: Progress | None = None,
    ) -> concurrent.futures.Future:
        """
        Run the simulation using multiple processes.

        Returns immediately with a Future that resolves to the aggregated
        result once all trials are complete.

        Parameters
        ----------
        summarizer : type[SimulationTables]
        output_dir : pathlib.Path, optional
            The directory to write output files to.  If not provided, a temporary
            directory will be created.
        fail_fast : bool, default True
            If True, if any subprocess fails by raising an exception, then cancel
            pending tasks and set the exception on the returned Future.
        run_id : str, optional
            An optional identifier for this run, which will be included in log
            messages and progress bars.  Set to `True` to generate a random id.
        jobber : JobExecutor, required
            A mp_executor.JobExecutor to which work will be submitted.

        Returns
        -------
        concurrent.futures.Future[SimulationTables]
            A Future that resolves to the aggregated simulation results.
        """
        run_start = time.time()
        run_start_str = datetime.fromtimestamp(run_start, UTC).isoformat()

        summarizer = check_summarizer(summarizer)

        # if no run_id is provided, the cfg.scenario is used.
        if run_id is None:
            run_id = self.config.scenario

        # if there is still no run_id, create a short random one.
        if not run_id:
            run_id = str(uuid.uuid4().hex[:4])

        # ensure run_id is a string and add a space on the end if not already present,
        # for nicer formatting in progress bars and logs
        run_id = str(run_id)
        if not run_id.endswith(" "):
            run_id = run_id + " "

        task_ids = list(range(self.config.simulation_controls.num_trials))
        num_samples = self.config.simulation_controls.num_samples
        cfg_json = self._dump_config()
        serialized_rm_systems = export_registered_rm_systems()

        dill_kwargs_blob = dill.dumps(
            {
                "summarizer": summarizer,
                "callbacks": self.callback_functions(),
                "exported_rm_systems": serialized_rm_systems,
            }
        )

        # Use a Manager queue so the proxy object is picklable with standard
        # pickle (required by ProcessPoolExecutor).
        manager = multiprocessing.Manager()
        progress_queue = manager.Queue()

        # Submit all trials to the executor
        trial_futures: dict[int, concurrent.futures.Future] = {}
        for task_id in task_ids:
            fut = jobber.executor.submit(
                _executor_run_trial,
                task_id,
                cfg_json,
                progress_queue,
                output_dir,
                dill_kwargs_blob,
            )
            trial_futures[task_id] = fut

        result_future: concurrent.futures.Future = concurrent.futures.Future()
        config = self.config

        # use provided rich_progress if available, otherwise use the one from the jobber
        if rich_progress is None:
            rich_progress = jobber.rich_progress

        def _monitor_and_finalize():
            try:
                task_progress_ids = {
                    task_id: rich_progress.add_task(f"[green]{run_id}Trial {task_id}", total=num_samples)
                    for task_id in task_ids
                }

                completed_tasks: set[int] = set()
                failed = False

                while len(completed_tasks) < len(task_ids) and not failed:
                    try:
                        task_id, message, payload = progress_queue.get(timeout=0.25)
                    except Exception:
                        # Queue empty or communication error – fall through
                        # to check whether any futures have resolved.
                        for tid, f in trial_futures.items():
                            if tid in completed_tasks or not f.done():
                                continue
                            exc = f.exception()
                            if exc is not None and fail_fast:
                                for ff in trial_futures.values():
                                    ff.cancel()
                                if not result_future.done():
                                    result_future.set_exception(exc)
                                failed = True
                                break
                            completed_tasks.add(tid)
                        continue

                    if message == "done":
                        rich_progress.update(
                            task_progress_ids[task_id],
                            completed=num_samples,
                            description=f"{run_id}Finished {task_id}",
                            refresh=True,
                            visible=False,
                        )
                        completed_tasks.add(task_id)
                    elif message == "update":
                        rich_progress.update(
                            task_progress_ids[task_id],
                            advance=payload,
                            refresh=True,
                        )
                    elif message == "finalizing":
                        rich_progress.update(
                            task_progress_ids[task_id],
                            description=f"{run_id}Finalize {task_id}",
                            refresh=True,
                        )
                    elif message == "exception":
                        # The ``payload`` is the exception raised inside the
                        # subprocess.  Because tracebacks cannot be pickled,
                        # ``payload.__traceback__`` is ``None`` here; the
                        # formatted traceback text was attached in the
                        # subprocess as ``_passengersim_subprocess_tb`` (and as
                        # a PEP 678 note).  Prefer printing that captured text
                        # so the user sees where the failure actually occurred.
                        sub_tb = getattr(payload, "_passengersim_subprocess_tb", None)
                        if sub_tb:
                            rich_progress.console.print(
                                f"[red]Exception in Trial {task_id} (subprocess traceback)[/red]:\n{sub_tb.rstrip()}"
                            )
                        else:
                            tb = _RichTraceback.from_exception(type(payload), payload, payload.__traceback__)
                            rich_progress.console.print(f"[red]Exception in Trial {task_id}[/red]:", tb)
                        completed_tasks.add(task_id)
                        if fail_fast:
                            for f in trial_futures.values():
                                f.cancel()
                            if not result_future.done():
                                result_future.set_exception(payload)
                            failed = True
                    else:
                        print(f"Unknown message {message}")

                if failed:
                    return

                # Aggregate results from the individual futures
                results: dict[int, object] = {}
                for task_id in task_ids:
                    try:
                        results[task_id] = trial_futures[task_id].result()
                    except Exception as e:
                        if fail_fast and not result_future.done():
                            result_future.set_exception(e)
                            return
                        # non-fail-fast: skip failed trials

                result = summarizer.aggregate([v for k, v in sorted(results.items())])
                result.config = config.model_copy(deep=True)
                result._metadata["time.started"] = run_start_str
                run_finished = time.time()
                result._metadata["time.runtime"] = run_finished - run_start
                result._metadata["time.finished"] = datetime.fromtimestamp(run_finished, UTC).isoformat()

                # write output files if designated
                output_filenames = {}
                output_filenames["html"] = config.outputs.get_output_filename(
                    "html",
                    timestamp=self._timestamp,
                )
                output_filenames["disk"] = config.outputs.get_output_filename(
                    "disk",
                    timestamp=self._timestamp,
                )
                if output_filenames["html"]:
                    result._metadata["outputs.html_filename"] = output_filenames["html"]
                if output_filenames["disk"]:
                    result._metadata["outputs.disk_filename"] = output_filenames["disk"]

                if output_filenames["html"]:
                    try:
                        result.to_html(output_filenames["html"], cfg=config, make_dirs=True, add_timestamp=False)
                    except Exception as err:
                        warnings.warn(f"Error writing HTML file: {err}", stacklevel=2)
                if output_filenames["disk"]:
                    try:
                        result.to_file(output_filenames["disk"], make_dirs=True, add_timestamp_ext=False)
                    except Exception as err:
                        warnings.warn(f"Error writing PXSIM file: {err}", stacklevel=2)

                result_future.set_result(result)

            except Exception as e:
                if not result_future.done():
                    result_future.set_exception(e)
            finally:
                try:
                    manager.shutdown()
                except Exception:
                    pass

        monitor_thread = threading.Thread(target=_monitor_and_finalize, daemon=True)
        monitor_thread.start()

        return result_future

    def run[SimulationTablesT: GenericSimulationTables](
        self,
        *,
        summarizer: type[SimulationTablesT] | SimulationTablesT | None = None,
        output_dir: pathlib.Path | None = None,
        max_processes: int | None = None,
        fail_fast: bool = True,
        rich_progress: Progress | None = None,
    ):
        summarizer = check_summarizer(summarizer)
        jobber = JobExecutor(max_workers=max_processes)

        with jobber:
            future = self._run_asynchronously(
                summarizer=summarizer,
                output_dir=output_dir,
                rich_progress=rich_progress,
                fail_fast=fail_fast,
                # executor=executor,
                jobber=jobber,
            )
            return future.result()
