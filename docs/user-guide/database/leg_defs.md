# Leg Definitions

The `leg_defs` database table stores static details about the legs in the
simulation.  Simulation results at the leg level are stored in the
[`leg_details`](leg_detail.md) table instead.

The `leg_defs` table is created by the
[create_table_leg_defs][passengersim.database.tables.create_table_leg_defs] function,
which is called in the [Simulation][passengersim.Simulation] initialization
step, so it should be available and populated for every simulation run.

## Table Schema

| Column    | Data Type           | Description                                        |
|:----------|:--------------------|:---------------------------------------------------|
| leg_id    | INTEGER PRIMARY KEY | Unique identifier for a given leg [^1]             |
| flt_no    | INTEGER             | Flight number label for this leg                   |
| carrier   | TEXT                | Name of carrier for this leg                       |
| orig      | TEXT                | Origin (typically an airport code or similar)      |
| dest      | TEXT                | Destination (typically an airport code or similar) |
| dep_time  | INTEGER             |                                                    |
| arr_time  | INTEGER             |                                                    |
| capacity  | INTEGER             | Number of seats on this leg                        |
| distance	 | FLOAT               | Distance from `orig` to `dest` in miles.           |


[^1]:
    In the "real world" the limitations of current technology make it such that
    flight numbers are not necessary unique by leg, as a single carrier may have
    multiple segments sharing the same flight number, and multiple carriers will
    have completely unrelated flights with the same flight number.  To simplify
    data processing, PassengerSim uses a unique id for every travel segment. Networks
    in PassengerSim that are derived from realistic sources can still store flight
    numbers as a nominal label for every leg, but they are not used for anything
    except certain post-simulation reporting features.
