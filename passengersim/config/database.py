import pathlib
from collections.abc import Callable
from typing import Literal

from pydantic import field_validator

from .pretty import PrettyModel


class DatabaseConfig(PrettyModel, extra="forbid", validate_assignment=True):
    engine: Literal["sqlite", "mysql", None] = "sqlite"
    """Database engine to use.

    Currently only `sqlite` is fully implemented."""

    filename: pathlib.Path | None = "passengersim-output.sqlite"
    """Name of file for SQLite output.

    When using the sqlite engine, an in-memory database can be created by
    setting this to ':memory:'.  Note that in-memory databases are not
    persistent and will be lost when the database connection is closed.  Also,
    in-memory databases cannot be shared between multiple processes.  When
    using multiprocessing, all database operations must be performed in the
    same process that created the database connection. Be sure to run summary
    queries before closing the database connection, by identifying all database
    summary tables in the `outputs.reports` configuration option.
    """

    fast: bool = False
    """Whether to use pre-compiled SQL write instructions."""

    pragmas: list[str] = []
    """A list of PRAGMA commands to execute upon opening a database connection."""

    commit_count_delay: int | None = 250
    """Commit transactions to the database will only be honored this frequently.

    By setting this number greater than 1, the transaction frequency will be reduced,
    improving overall runtime performance by storing more data in RAM and writing to
    persistent storage less frequently.
    """

    write_items: set[
        Literal[
            "leg",
            "bucket",
            "edgar",
            "fare",
            "demand",
            "leg_daily",
            "leg_final",
            "bucket_final",
            "fare_final",
            "demand_final",
            "bookings",
            "pathclass",
            "pathclass_daily",
            "pathclass_final",
        ]
    ] = {"leg_final", "fare_final", "demand_final", "bookings"}
    """Which items should be written to the database.

    The following values can be provided in this set:

    - *leg*: write every leg to the `leg_detail` table at every DCP.
    - *leg_final*: write every leg to the `leg_detail` table only at DCP 0.
    - *bucket*: write every leg bucket to the `leg_bucket_detail` table at every DCP.
    - *bucket_final*: write every leg bucket to the `leg_bucket_detail` table only
        at DCP 0.
    - *fare*: write every fare to the `fare_detail` table at every DCP.
    - *fare_final*: write every fare to the `fare_detail` table only at DCP 0.
    - *demand*: write every demand to the `demand_detail` table at every DCP.
    - *bookings*: store booking summary data at every DCP and write an aggregate
        summary of bookings by DCP to the `bookings_by_timeframe` table at the end
        of the simulation.
    """

    dcp_write_hooks: list[Callable] = []
    """Additional callable functions that write to the database at each DCP.

    Each should have a signature matching `f(db, sim, dcp)`.
    """

    @field_validator("engine", mode="before")
    def _interpret_none(cls, v):
        """Allow engine to be "none"."""
        if isinstance(v, str) and v.lower() == "none":
            v = None
        return v

    store_leg_bid_prices: bool = True
    """Should leg bid prices be stored in the database."""

    store_displacements: bool = True
    """Should leg displacements be stored in the database."""
