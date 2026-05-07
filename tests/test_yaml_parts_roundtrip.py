"""Test round-trip of passengersim.Config through YAML serialization via to_yaml_parts.

This test exercises all features of the ``human_readable`` serialization mode,
which writes leg departure and arrival times as 'HH:MM' strings in the local
time zone at each airport, and records a non-zero ``arr_day`` offset when a
leg crosses midnight (in local time) between origin and destination.

Legs included to exercise every case:

- **NRT → SEA** (flight 1001): departs Tokyo (Asia/Tokyo) on local March 2 at
  01:00 JST, arrives Seattle (America/Los_Angeles) on March 1 at 17:38 PST.
  The local departure date at NRT is March 2, but the local arrival date at SEA
  is March 1 — a classic "gain a day" crossing when flying eastbound across the
  International Date Line.  ``arr_day = -1``.

- **SEA → FCO** (flight 1002): departs Seattle (America/Los_Angeles) on March 1
  at 14:00 PST, arrives Rome (Europe/Rome) on March 2 at 10:00 CET — an
  overnight transatlantic flight.  ``arr_day = +1``.

- **SEA → LAX** (flight 1003): same-day domestic leg (Seattle to Los Angeles,
  ``arr_day = 0``).  Verifies that ``arr_day`` is *omitted* from the human-
  readable YAML when it is zero.
"""

import pathlib

import pytest
import yaml

from passengersim.config import Config

# The FCFS RM system lives in the "specialty_systems" sub-package and must be
# explicitly imported to be registered before Config.model_validate is called.
from passengersim.rm.specialty_systems.fcfs import FirstComeFirstServed  # noqa: F401

# ---------------------------------------------------------------------------
# Minimal YAML configuration with international legs
# ---------------------------------------------------------------------------

CONFIG_YAML = """\
scenario: test_intl_roundtrip

simulation_controls:
  num_samples: 10
  burn_samples: 5
  random_seed: 42

# Set db to null so we do not need a database to validate the config.
db: null

carriers:
  - name: AL1
    rm_system: FCFS

# Places supply the lat/lon and time-zone information used to
# (a) compute great-circle distances automatically, and
# (b) convert local HH:MM strings to UTC Unix timestamps during validation.
places:
  NRT:
    label: Tokyo Narita International Airport
    lat: 35.7647
    lon: 140.3864
    time_zone: Asia/Tokyo
  SEA:
    label: Seattle-Tacoma International Airport
    lat: 47.4502
    lon: -122.3088
    time_zone: America/Los_Angeles
  FCO:
    label: Rome Fiumicino Airport
    lat: 41.8003
    lon: 12.2389
    time_zone: Europe/Rome
  LAX:
    label: Los Angeles International Airport
    lat: 33.9438
    lon: -118.4091
    time_zone: America/Los_Angeles

legs:
  # NRT→SEA: crosses the International Date Line eastbound.
  # Departs Tokyo (Asia/Tokyo) on local March 2 at 01:00 JST = March 1 16:00 UTC.
  # Arrives Seattle (America/Los_Angeles) ~9 h 38 min later at
  #   March 2 01:38 UTC = March 1 17:38 PST.
  # Local departure date (NRT) = March 2; local arrival date (SEA) = March 1.
  # arr_day = -1: the destination calendar day is one behind the origin calendar day.
  - carrier: AL1
    fltno: 1001
    orig: NRT
    dest: SEA
    date: '2020-03-02'
    dep_time: '01:00'
    arr_time: '17:38'
    arr_day: -1
    capacity: 250
    tags:
      route_type: transpacific

  # SEA→FCO: transatlantic overnight flight.
  # Departs Seattle (America/Los_Angeles) March 1 at 14:00 PST = March 1 22:00 UTC.
  # Arrives Rome (Europe/Rome) exactly 11 hours later at
  #   March 2 09:00 UTC = March 2 10:00 CET.
  # Local departure date (SEA) = March 1; local arrival date (FCO) = March 2.
  # arr_day = +1: the destination calendar day is one ahead of the origin calendar day.
  - carrier: AL1
    fltno: 1002
    orig: SEA
    dest: FCO
    date: '2020-03-01'
    dep_time: '14:00'
    arr_time: '10:00'
    arr_day: 1
    capacity: 250
    tags:
      route_type: transatlantic

  # SEA→LAX: same-day domestic leg.
  # Departs Seattle 08:00 PST, arrives Los Angeles 10:30 PST (~2.5 h flight).
  # arr_day = 0 (default) — departure and arrival on the same local calendar day.
  - carrier: AL1
    fltno: 1003
    orig: SEA
    dest: LAX
    date: '2020-03-01'
    dep_time: '08:00'
    arr_time: '10:30'
    capacity: 180
"""


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def intl_config() -> Config:
    """Build a PassengerSim Config containing international and domestic legs."""
    return Config.from_raw_yaml(CONFIG_YAML)


# ---------------------------------------------------------------------------
# Sanity-check the fixture itself
# ---------------------------------------------------------------------------


def test_intl_config_leg_local_times(intl_config: Config):
    """Verify the loaded legs carry the expected local departure/arrival times.

    This checks that Config validation correctly converts 'HH:MM' strings
    (with arr_day offsets) to UTC Unix timestamps via the time-zone adjustment.
    """
    assert len(intl_config.legs) == 3

    by_fltno = {leg.fltno: leg for leg in intl_config.legs}

    # --- NRT→SEA (arr_day = -1) ---
    nrt_sea = by_fltno[1001]
    assert nrt_sea.orig_timezone == "Asia/Tokyo"
    assert nrt_sea.dest_timezone == "America/Los_Angeles"

    # Departure: 2020-03-02 01:00 JST
    dep = nrt_sea.dep_localtime
    assert (dep.year, dep.month, dep.day) == (2020, 3, 2)
    assert (dep.hour, dep.minute) == (1, 0)

    # Arrival: 2020-03-01 17:38 PST
    arr = nrt_sea.arr_localtime
    assert (arr.year, arr.month, arr.day) == (2020, 3, 1)
    assert (arr.hour, arr.minute) == (17, 38)

    # Duration: 9 h 38 min = 578 min
    assert nrt_sea.duration_minutes == 578

    # tags survive the initial parse
    assert nrt_sea.tags.get("route_type") == "transpacific"

    # --- SEA→FCO (arr_day = +1) ---
    sea_fco = by_fltno[1002]
    assert sea_fco.orig_timezone == "America/Los_Angeles"
    assert sea_fco.dest_timezone == "Europe/Rome"

    # Departure: 2020-03-01 14:00 PST
    dep2 = sea_fco.dep_localtime
    assert (dep2.year, dep2.month, dep2.day) == (2020, 3, 1)
    assert (dep2.hour, dep2.minute) == (14, 0)

    # Arrival: 2020-03-02 10:00 CET
    arr2 = sea_fco.arr_localtime
    assert (arr2.year, arr2.month, arr2.day) == (2020, 3, 2)
    assert (arr2.hour, arr2.minute) == (10, 0)

    # Duration: exactly 11 h = 660 min
    assert sea_fco.duration_minutes == 660

    assert sea_fco.tags.get("route_type") == "transatlantic"

    # --- SEA→LAX (arr_day = 0, same-day domestic) ---
    sea_lax = by_fltno[1003]

    dep3 = sea_lax.dep_localtime
    arr3 = sea_lax.arr_localtime
    assert (dep3.hour, dep3.minute) == (8, 0)
    assert (arr3.hour, arr3.minute) == (10, 30)
    # Both local times fall on the same calendar day
    assert dep3.date() == arr3.date()


# ---------------------------------------------------------------------------
# Round-trip tests
# ---------------------------------------------------------------------------


def test_to_yaml_parts_roundtrip_human_readable(intl_config: Config, tmp_path: pathlib.Path):
    """Round-trip a Config via to_yaml_parts(human_readable=True) / from_yaml.

    Verifies that leg departure/arrival times expressed in local-time 'HH:MM'
    format (including non-zero arr_day offsets for the international legs)
    survive serialization and deserialization unchanged.

    Notes
    -----
    The human-readable leg serializer writes distances rounded to 3 decimal
    places (``round(distance, 3)``).  This is an intentional formatting choice
    and introduces a small, bounded precision loss on the *first* round-trip
    when distances were originally auto-computed to full floating-point precision
    from place lat/lon.  The test therefore:

    1. Verifies that the *only* differences after the first round-trip are in
       leg distances and that those differences are within the rounding tolerance
       (< 0.001 miles / ~1.6 m).
    2. Verifies that subsequent round-trips are fully idempotent (the rounded
       distance is already at 3 dp, so no further precision is lost).
    """
    out_dir = tmp_path / "yaml_parts_hr"

    # Serialize: writes legs.yaml, places.yaml, carriers.yaml, general.yaml,
    # outputs.yaml, and an __main__.yaml index file.
    intl_config.to_yaml_parts(out_dir, human_readable=True)

    main_yaml = out_dir / "__main__.yaml"
    assert main_yaml.exists(), "__main__.yaml was not created by to_yaml_parts"

    # Re-load from the serialized YAML parts
    cfg2 = Config.from_yaml(main_yaml)

    # find_differences compares model_dump() outputs (raw Unix timestamps) and
    # excludes raw_license_certificate and a handful of output-file paths.
    diff = intl_config.find_differences(cfg2)

    # The human_readable serializer rounds leg distances to 3 decimal places.
    # Strip those from the diff before asserting — and separately verify that
    # the differences are within the expected rounding bound of 0.001 miles.
    if "legs" in diff:
        for leg_diff in dict(diff["legs"]).values():
            if isinstance(leg_diff, dict):
                leg_diff.pop("distance", None)
        diff["legs"] = {k: v for k, v in diff["legs"].items() if v}
        if not diff["legs"]:
            del diff["legs"]

    assert diff == {}, f"Round-trip (human_readable=True) found unexpected differences:\n{diff}"

    # Verify: the only distance loss is at most half a unit in the 3rd decimal
    # place (i.e., < 0.001 miles ≈ 1.6 m), which is the precision of the
    # human-readable distance format.
    for leg1, leg2 in zip(intl_config.legs, cfg2.legs):
        if leg1.distance is not None and leg2.distance is not None:
            assert abs(leg1.distance - leg2.distance) < 0.001, (
                f"Distance for {leg1.orig}→{leg1.dest} changed by more than "
                f"rounding tolerance: {leg1.distance} vs {leg2.distance}"
            )

    # Verify idempotency: a second round-trip from cfg2 (which already stores
    # the rounded distances) must be completely lossless.
    out_dir2 = tmp_path / "yaml_parts_hr2"
    cfg2.to_yaml_parts(out_dir2, human_readable=True)
    cfg3 = Config.from_yaml(out_dir2 / "__main__.yaml")
    diff2 = cfg2.find_differences(cfg3)
    assert diff2 == {}, f"Second round-trip (human_readable=True) is not idempotent:\n{diff2}"


def test_to_yaml_parts_roundtrip_raw(intl_config: Config, tmp_path: pathlib.Path):
    """Round-trip a Config via to_yaml_parts(human_readable=False) / from_yaml.

    Confirms that the raw Unix-timestamp mode also produces an identical Config
    upon reload, providing a baseline comparison to the human-readable mode.
    """
    out_dir = tmp_path / "yaml_parts_raw"

    intl_config.to_yaml_parts(out_dir, human_readable=False)

    main_yaml = out_dir / "__main__.yaml"
    assert main_yaml.exists(), "__main__.yaml was not created by to_yaml_parts"

    cfg2 = Config.from_yaml(main_yaml)

    diff = intl_config.find_differences(cfg2)
    assert diff == {}, f"Round-trip (human_readable=False) found differences:\n{diff}"


# ---------------------------------------------------------------------------
# Inspect the human-readable YAML content directly
# ---------------------------------------------------------------------------


def test_yaml_parts_human_readable_leg_content(intl_config: Config, tmp_path: pathlib.Path):
    """Inspect the legs.yaml produced by to_yaml_parts(human_readable=True).

    Asserts the exact values written for each leg in the human-readable
    serialization format, verifying that:

    * dep_time and arr_time are written as 'HH:MM' strings in local time.
    * orig_timezone and dest_timezone are explicitly recorded.
    * arr_day is present (and correct) for the two international legs.
    * arr_day is absent for the same-day domestic leg.
    * leg tags are written unchanged.
    """
    out_dir = tmp_path / "yaml_parts_content"
    intl_config.to_yaml_parts(out_dir, human_readable=True)

    legs_yaml = out_dir / "legs.yaml"
    assert legs_yaml.exists(), "legs.yaml was not written"

    with open(legs_yaml) as fh:
        raw = yaml.safe_load(fh)

    leg_list = raw["legs"]
    assert isinstance(leg_list, list)
    by_fltno = {leg["fltno"]: leg for leg in leg_list}

    # --- NRT→SEA (flight 1001, arr_day = -1) ---
    nrt_sea = by_fltno[1001]
    assert nrt_sea["orig"] == "NRT"
    assert nrt_sea["dest"] == "SEA"

    # Time-zone annotations must appear in the YAML
    assert nrt_sea["orig_timezone"] == "Asia/Tokyo"
    assert nrt_sea["dest_timezone"] == "America/Los_Angeles"

    # dep_time is in local 24-h time at NRT; arr_time is in local time at SEA
    assert nrt_sea["dep_time"] == "01:00"
    assert nrt_sea["arr_time"] == "17:38"

    # arr_day = -1 serialized as a "+d"-format string (e.g. "-1"); must round-trip
    # to int -1 via pydantic coercion.
    assert int(nrt_sea["arr_day"]) == -1

    # Tags must survive the round-trip
    assert nrt_sea.get("tags", {}).get("route_type") == "transpacific"

    # --- SEA→FCO (flight 1002, arr_day = +1) ---
    sea_fco = by_fltno[1002]
    assert sea_fco["orig"] == "SEA"
    assert sea_fco["dest"] == "FCO"

    assert sea_fco["orig_timezone"] == "America/Los_Angeles"
    assert sea_fco["dest_timezone"] == "Europe/Rome"

    assert sea_fco["dep_time"] == "14:00"
    assert sea_fco["arr_time"] == "10:00"

    assert int(sea_fco["arr_day"]) == 1

    assert sea_fco.get("tags", {}).get("route_type") == "transatlantic"

    # --- SEA→LAX (flight 1003, arr_day = 0) ---
    sea_lax = by_fltno[1003]
    assert sea_lax["orig"] == "SEA"
    assert sea_lax["dest"] == "LAX"

    assert sea_lax["dep_time"] == "08:00"
    assert sea_lax["arr_time"] == "10:30"

    # arr_day must NOT appear in the YAML when it equals zero
    assert "arr_day" not in sea_lax, "arr_day should be omitted from human-readable YAML when it is zero"


def test_yaml_parts_include_file_structure(intl_config: Config, tmp_path: pathlib.Path):
    """Check the directory structure written by to_yaml_parts.

    The __main__.yaml index file should list the individual part files, and
    each part file must actually exist on disk.
    """
    out_dir = tmp_path / "yaml_parts_structure"
    intl_config.to_yaml_parts(out_dir, human_readable=True)

    main_yaml = out_dir / "__main__.yaml"
    assert main_yaml.exists()

    with open(main_yaml) as fh:
        main_data = yaml.safe_load(fh)

    assert "include" in main_data, "__main__.yaml must have an 'include' key"
    included_files = main_data["include"]
    assert isinstance(included_files, list)
    assert len(included_files) > 0, "to_yaml_parts must write at least one part file"

    # Every file listed in __main__.yaml must exist
    for fname in included_files:
        part_path = out_dir / fname
        assert part_path.exists(), f"Included file {fname!r} was not found on disk"

    # legs.yaml must be among the written parts
    assert "legs.yaml" in included_files, "legs.yaml should be written as a separate part"

    # places.yaml must also be present (needed for time-zone information)
    assert "places.yaml" in included_files, "places.yaml should be written as a separate part"
