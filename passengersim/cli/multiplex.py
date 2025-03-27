#  Based on https://www.deanmontgomery.com/2022/03/24/rich-progress-and-multiprocessing/


import multiprocessing
import os
import pathlib
import time
from concurrent.futures import ProcessPoolExecutor

import rich.progress

from passengersim._logging import log_to_file
from passengersim.cli._app import app
from passengersim.config import Config
from passengersim.config.base import YamlConfig
from passengersim.driver import Simulation


def run_task(progress, task_id, job_name: str, config_files: list[str | pathlib.Path]):
    def callback(n, n_total):
        progress[task_id] = {"progress": n, "total": n_total}

    callback(0, None)
    log_file_name = job_name.replace(" ", "-") + ".log"
    # with open(log_file_name, 'w') as sys.stdout:
    _logger = log_to_file(log_file_name)
    try:
        cfg = Config.from_yaml(config_files)
        cfg.simulation_controls.show_progress_bar = False
        sim = Simulation(cfg)
        sim.sample_done_callback = callback
        callback(
            0, cfg.simulation_controls.num_trials * cfg.simulation_controls.num_samples
        )
        sim.run(log_reports=False)
    except Exception:
        _logger.exception(f"Error in TASK {task_id}")


class MultiplexConfig(YamlConfig, extra="forbid"):
    n_workers: int = -1
    jobs: dict[str, pathlib.Path | list[pathlib.Path]]
    working_dir: pathlib.Path = "."
    spool: bool = True
    """Generate a spooled subdirectory for run results."""

    spool_format: str = "%Y%m%d-%H%M"
    """Format for creating spooled subdirectory."""


@app.command()
def multi(config_file: pathlib.Path):
    mp = MultiplexConfig.from_yaml([config_file])

    n_workers = mp.n_workers
    if n_workers <= 0:
        n_workers = max(multiprocessing.cpu_count() - n_workers, 1)

    if mp.working_dir.is_absolute():
        working_dir = mp.working_dir
    else:
        working_dir = config_file.parent.joinpath(mp.working_dir)

    if mp.spool:
        proposal = working_dir.joinpath(time.strftime(mp.spool_format))
        n = 0
        while proposal.exists():
            n += 1
            proposal = working_dir.joinpath(time.strftime(mp.spool_format) + f".{n}")
        working_dir = proposal
        working_dir.mkdir(parents=True)
        # now prefix all relative job file locations with ..
        jobs = {}
        for (
            job_name,
            orig_job,
        ) in mp.jobs.items():
            job = []
            for oj in orig_job:
                if str(oj) == "SKIP":
                    job.append("SKIP")
                elif os.path.isabs(oj):
                    job.append(oj)
                else:
                    job.append(pathlib.Path("..").joinpath(oj))
            jobs[job_name] = job
    else:
        jobs = mp.jobs

    os.chdir(working_dir)

    with rich.progress.Progress(
        "[progress.description]{task.description}",
        rich.progress.BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        rich.progress.TimeRemainingColumn(),
        rich.progress.TimeElapsedColumn(),
        refresh_per_second=1,  # bit slower updates
    ) as progress:
        futures = []  # keep track of the jobs
        with multiprocessing.Manager() as manager:
            # this is the key - we share some state between our
            # main process and our worker functions
            _progress = manager.dict()
            overall_progress_task = progress.add_task("[green]JOBS")

            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                for (
                    job_name,
                    job,
                ) in jobs.items():  # iterate over the jobs we need to run
                    if "SKIP" in job:
                        continue
                    # set visible false so we don't have a lot of bars all at once:
                    task_id = progress.add_task(job_name, visible=False)
                    futures.append(
                        executor.submit(run_task, _progress, task_id, job_name, job)
                    )

                # monitor the progress:
                while (n_finished := sum([future.done() for future in futures])) < len(
                    futures
                ):
                    progress.update(
                        overall_progress_task, completed=n_finished, total=len(futures)
                    )
                    for task_id, update_data in _progress.items():
                        latest = update_data["progress"]
                        total = update_data["total"]
                        # update the progress bar for this task:
                        progress.update(
                            task_id,
                            completed=latest,
                            total=total,
                            visible=total is None or (latest < total),
                        )

                # raise any errors:
                for future in futures:
                    future.result()

            # final overall progress update
            n_finished = sum([future.done() for future in futures])
            progress.update(
                overall_progress_task, completed=n_finished, total=len(futures)
            )
