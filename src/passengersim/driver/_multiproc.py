from __future__ import annotations

import concurrent.futures
import getpass
import json
import logging
import multiprocessing
import pathlib
import queue
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

from passengersim.callbacks import CallbackMixin
from passengersim.config import Config
from passengersim.core import SimulationEngine
from passengersim.driver._base_sim import BaseSimulation
from passengersim.driver._singleproc import Simulation, check_summarizer
from passengersim.driver._trial_database import TrialDatabaseStore, TrialResult, finalize_result, finish_database
from passengersim.mp_executor import JobExecutor
from passengersim.rm.systems import export_registered_rm_systems, restore_registered_rm_systems
from passengersim.summaries import GenericSimulationTables, SimulationTables
from passengersim.utils.tempdir import MaybeTemporaryDirectory

if typing.TYPE_CHECKING:
    from rich.progress import Progress


class ThrottledUpdater:
    def __init__(self, progress_queue, trial_id, throttle: float = 0.5):
        self.progress_queue = progress_queue
        self.trial_id = trial_id
        self.throttle = throttle
        self.last_time = 0
        self.n = 0
        self.completed = 0

    def __call__(self, completed, total):
        # The callback includes an initial zero; count samples, not callbacks.
        self.n += completed - self.completed
        self.completed = completed
        if self.n and time.time() - self.last_time > self.throttle:
            self.flush()

    def flush(self):
        self.progress_queue.put((self.trial_id, "update", self.n))
        self.last_time = time.time()
        self.n = 0


def _subprocess_run_trial(
    trial_id: int,
    cfg_json: str,
    progress_queue: multiprocessing.Queue | None = None,
    output_dir: pathlib.Path | None = None,
    *,
    summarizer=SimulationTables,
    callbacks=None,
    exported_rm_systems=None,
    database_path: pathlib.Path | None = None,
):
    trial_started = time.perf_counter()
    sim = None
    if progress_queue is None:
        progress_queue = queue.SimpleQueue()
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
        if database_path is not None:
            cfg.db.filename = database_path
            cfg.db.result_directory = None
            cfg.db.result_storage = "persistent"
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
        simulation_finished = time.perf_counter()
        sim.sample_done_callback.flush()
        progress_queue.put((trial_id, "finalizing", None))
        # Passing a database connection between processes is not allowed,
        # so we need to delete it before returning the summary. But first
        # we will run any queries that were requested as output reports
        if isinstance(summary, GenericSimulationTables) and summary.db is not None:
            summary.run_queries(items=cfg.outputs.reports)
        summary.__dict__.pop("_trial_database_store", None)
        try:
            summary.db = None
        except AttributeError as err:
            print(err)
        reports_finished = time.perf_counter()
        descriptor = None
        if database_path is not None:
            descriptor = finish_database(sim.db, trial_id)
        else:
            sim.db.close()
        progress_queue.put((trial_id, "done", None))
        published = time.perf_counter()
        return TrialResult(
            summary,
            descriptor,
            {
                "simulation_and_writes_s": simulation_finished - trial_started,
                "eager_reports_s": reports_finished - simulation_finished,
                "worker_finalization_s": published - reports_finished,
            },
            published,
        )
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
        progress_queue.put((trial_id, "exception", None))
        return e
    finally:
        if sim is not None and sim.db.connection is not None:
            # Failed trials are never advertised as complete. Roll back pending
            # work and release handles before the parent can clean up files.
            sim.db.disconnect_from_simulation_engine()
            sim.db.connection.rollback()
            sim.db.close()


def _executor_run_trial(
    trial_id: int,
    cfg_json: str,
    progress_queue,
    output_dir: pathlib.Path | None,
    dill_kwargs_blob: bytes,
    database_path: pathlib.Path | None = None,
):
    """Worker function for use with ProcessPoolExecutor.

    Deserializes dill-encoded kwargs and delegates to _subprocess_run_trial.
    Exceptions are re-raised so the Future captures them.
    """
    kwargs = dill.loads(dill_kwargs_blob)
    result = _subprocess_run_trial(
        trial_id, cfg_json, progress_queue, output_dir, database_path=database_path, **kwargs
    )
    if isinstance(result, Exception):
        raise result
    return result


class MultiSimulation(BaseSimulation, CallbackMixin):
    def __init__(
        self,
        config: Config,
        output_dir: pathlib.Path | None = None,
        *,
        database_initializer=None,
    ):
        self._database_initializer = database_initializer
        super().__init__(config, output_dir)
        self.config = config
        self._placeholder_sim = None

    def close(self):
        """Close the current result and every database alias; preserve persistent files."""
        store = getattr(self, "_trial_database_store", None)
        if store is not None:
            store.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    @property
    def db(self):
        """The shared consolidated database, opened on the consuming thread."""
        store = getattr(self, "_trial_database_store", None)
        return store.database if store is not None else None

    @db.setter
    def db(self, value):
        # BaseSimulation initializes this attribute before a run has an owner.
        if value is not None:
            raise AttributeError("MultiSimulation.db is managed by its result owner")

    @property
    def database_path(self):
        """Absolute consolidated result path, without opening its connection."""
        store = getattr(self, "_trial_database_store", None)
        return store.consolidated_path if store is not None else None

    @property
    def database_available(self):
        """Whether this simulation has retained detail data."""
        store = getattr(self, "_trial_database_store", None)
        return bool(store is not None and store.available)

    @property
    def database_status(self):
        """Completed, failed, and cancelled trials, plus whole-run completeness."""
        store = getattr(self, "_trial_database_store", None)
        return store.status if store is not None else None

    @property
    def trial_databases(self):
        """Completed trial descriptors; temporary trials share the consolidated path."""
        store = getattr(self, "_trial_database_store", None)
        return dict(store.completed) if store is not None else {}

    def sequential_run(self, *, summarizer=SimulationTables, fail_fast=True):
        """Run trials sequentially using the same closed-file handoff as workers."""
        run_start = time.time()
        summarizer = check_summarizer(summarizer)
        from passengersim.config.storage import resolve_storage, validate_database_reports

        validate_database_reports(self.config, resolve_storage(self.config.db, orchestrated=True), summarizer)
        store = self._trial_database_store = TrialDatabaseStore(self.config, initializer=self._database_initializer)
        results = []
        cfg_json = self._dump_config()
        for trial_id in range(self.config.simulation_controls.num_trials):
            try:
                value = _subprocess_run_trial(
                    trial_id,
                    cfg_json,
                    output_dir=self.output_dir,
                    summarizer=summarizer,
                    callbacks=self.callback_functions(),
                    database_path=store.path_for(trial_id),
                )
                if isinstance(value, Exception):
                    raise value
                store.accept(trial_id, value)
                results.append(value.summary)
            except BaseException as error:
                store.failed[trial_id] = str(error)
                if fail_fast:
                    store.cancelled.update(store.expected - store.merged - store.failed.keys())
                    store.finalized = True
                    raise
        store.finalized = True
        result = summarizer.aggregate(results)
        self._finalize_trial_summaries(result, self.config, run_start, store)
        result._trial_database_store = store
        return finalize_result(result)

    def _finalize_trial_summaries(self, result, config, run_start, store):
        """Preserve parent output/report behavior for both trial execution modes."""
        result._metadata["database.timings"] = store.metrics
        result.config = config.model_copy(deep=True)
        result._metadata["time.started"] = datetime.fromtimestamp(run_start, UTC).isoformat()
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
                result.to_file(
                    output_filenames["disk"],
                    make_dirs=True,
                    add_timestamp_ext=False,
                )
            except Exception as err:
                warnings.warn(f"Error writing PXSIM file: {err}", stacklevel=2)
        if filename := config.outputs.get_output_filename("pickle", timestamp=self._timestamp):
            result._metadata["outputs.pickle_filename"] = result.to_pickle(filename)
        if filename := config.outputs.get_output_filename("excel", timestamp=self._timestamp):
            result.to_xlsx(filename)
        return result

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

        summarizer = check_summarizer(summarizer)
        from passengersim.config.storage import resolve_storage, validate_database_reports

        validate_database_reports(self.config, resolve_storage(self.config.db, orchestrated=True), summarizer)

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
        try:
            progress_queue = manager.Queue()
            # Reserve storage only after serialization and manager setup succeed;
            # failures before worker submission must not consume a destination.
            store = self._trial_database_store = TrialDatabaseStore(self.config, initializer=self._database_initializer)
        except BaseException:
            manager.shutdown()
            raise
        store._running = True

        # Submit all trials to the executor
        trial_futures: dict[int, concurrent.futures.Future] = {}
        try:
            for task_id in task_ids:
                fut = jobber.executor.submit(
                    _executor_run_trial,
                    task_id,
                    cfg_json,
                    progress_queue,
                    output_dir,
                    dill_kwargs_blob,
                    store.path_for(task_id),
                )
                store.track(fut)
                trial_futures[task_id] = fut
        except BaseException:
            store.interrupted = True
            # Submission itself may fail after earlier workers have started.
            # Keep their progress queue and files alive until they finish.
            for future in trial_futures.values():
                future.cancel()
            concurrent.futures.wait(trial_futures.values())
            store.wait()
            for tid in task_ids:
                future = trial_futures.get(tid)
                if future is None or future.cancelled():
                    store.cancelled.add(tid)
                    continue
                try:
                    value = future.result()
                    store.accept(tid, value)
                except BaseException as error:
                    store.failed[tid] = str(error)
            store.finalized = True
            store._running = False
            manager.shutdown()
            raise

        result_future: concurrent.futures.Future = concurrent.futures.Future()
        config = self.config

        # use provided rich_progress if available, otherwise use the one from the jobber
        if rich_progress is None:
            rich_progress = jobber.rich_progress

        def _monitor_and_finalize():
            results = {}
            first_error = None
            pending = set(task_ids)
            task_progress_ids = {}
            try:
                for task_id in task_ids:
                    task_progress_ids[task_id] = rich_progress.add_task(
                        f"[green]{run_id}Trial {task_id}", total=num_samples
                    )
                while pending:
                    if result_future.cancelled():
                        store.interrupted = True
                    if result_future.cancelled() or (first_error is not None and fail_fast):
                        for tid in pending:
                            trial_futures[tid].cancel()
                    # Only the monitor merges. Inspect Futures independently of
                    # queue messages, which may precede actual file publication.
                    for tid in sorted(pending.copy()):
                        future = trial_futures[tid]
                        if not future.done():
                            continue
                        pending.remove(tid)
                        if future.cancelled():
                            store.cancelled.add(tid)
                            continue
                        try:
                            value = future.result()
                            store.accept(tid, value)
                            results[tid] = value.summary
                            # Futures are authoritative; final queue updates may be unread.
                            rich_progress.update(
                                task_progress_ids[tid],
                                completed=num_samples,
                                description=f"{run_id}Finished {tid}",
                            )
                        except BaseException as error:
                            store.failed[tid] = str(error)
                            if first_error is None:
                                first_error = error
                            # Worker traceback notes survive Future transport.
                            sub_tb = getattr(error, "_passengersim_subprocess_tb", None)
                            rich_progress.console.print(
                                f"Exception in Trial {tid}: {sub_tb or error}", style="red", markup=False
                            )
                        rich_progress.update(task_progress_ids[tid], visible=False, refresh=True)
                    if pending:
                        try:
                            tid, message, payload = progress_queue.get(timeout=0.05)
                        except queue.Empty:
                            continue
                        if tid not in pending:
                            continue  # Ignore late messages from finalized trials.
                        if message == "update":
                            rich_progress.update(task_progress_ids[tid], advance=payload, refresh=True)
                        elif message == "finalizing":
                            rich_progress.update(task_progress_ids[tid], description=f"{run_id}Finalize {tid}")
                store.wait()
                store.finalized = True
                if result_future.cancelled():
                    return
                if first_error is not None and fail_fast:
                    result_future.set_exception(first_error)
                    return
                result = summarizer.aggregate([v for k, v in sorted(results.items())])
                self._finalize_trial_summaries(result, config, run_start, store)
                # Publish the owner only after monitor-thread output generation;
                # no retained connection may be opened on this thread.
                result._trial_database_store = store
                result_future.set_result(result)
            except BaseException as error:
                store.interrupted = True
                for tid in pending:
                    trial_futures[tid].cancel()
                if not result_future.done():
                    result_future.set_exception(error)
            finally:
                # Cancellation cannot stop a running worker. Keep its queue and
                # storage alive until all file users and callbacks finish.
                concurrent.futures.wait(trial_futures.values())
                store.wait()
                # Even a monitor/output failure must preserve successful workers
                # that were still running when normal collection was interrupted.
                for tid in sorted(pending):
                    future = trial_futures[tid]
                    if future.cancelled():
                        store.cancelled.add(tid)
                        continue
                    try:
                        value = future.result()
                        store.accept(tid, value)
                    except BaseException as error:
                        store.failed[tid] = str(error)
                if result_future.cancelled():
                    store.interrupted = True
                store.finalized = True
                store._running = False
                try:
                    # Clear only this run's tasks, including failed/cancelled trials.
                    for progress_id in task_progress_ids.values():
                        rich_progress.remove_task(progress_id)
                    rich_progress.refresh()
                finally:
                    try:
                        manager.shutdown()
                    except Exception:
                        pass

        monitor_thread = threading.Thread(target=_monitor_and_finalize, daemon=True)
        result_future._monitor_thread = monitor_thread
        monitor_thread.start()

        return result_future

    def log_config(self):
        user_name = getpass.getuser()
        trials = self.config.simulation_controls.num_trials
        samples = self.config.simulation_controls.num_samples
        msg = f"user={user_name}, trials={trials}, samples={samples}"
        # Send a basic informational log message
        try:
            import syslog
        except ImportError:
            logging.getLogger(__name__).info(msg)
        else:
            syslog.openlog(ident="PassegerSim", facility=syslog.LOG_USER)
            syslog.syslog(syslog.LOG_INFO, msg)
            syslog.closelog()

    def run[SimulationTablesT: GenericSimulationTables](
        self,
        *,
        summarizer: type[SimulationTablesT] | SimulationTablesT | None = None,
        output_dir: pathlib.Path | None = None,
        max_processes: int | None = None,
        fail_fast: bool = True,
        rich_progress: Progress | None = None,
        cache_dir: pathlib.Path | None = None,
    ):
        summarizer = check_summarizer(summarizer)
        from passengersim.config.storage import resolve_storage, validate_database_reports

        validate_database_reports(self.config, resolve_storage(self.config.db, orchestrated=True), summarizer)
        # self.log_config()

        if cache_dir is not None:
            from ._cache_run import cache_run

            return cache_run(
                self,
                cache_dir=cache_dir,
                summarizer=summarizer,
                output_dir=output_dir,
                max_processes=max_processes,
                fail_fast=fail_fast,
                rich_progress=rich_progress,
            )

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
            try:
                result = future.result()
                future._monitor_thread.join()
                return finalize_result(result)
            finally:
                # Synchronous completion includes releasing worker queues and
                # monitor references, not merely publishing the summary.
                future._monitor_thread.join()
