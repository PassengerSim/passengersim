# Fare Definitions

The `fare_defs` database table stores static details about the fares in the
simulation.  Simulation results at the fare level are stored in the
[`fare_details`](fare_detail.md) table instead.

The `fare_defs` table is created by the
[create_table_fare_defs][passengersim.database.tables.create_table_fare_defs] function,
which is called in the [Simulation][passengersim.Simulation] initialization
step, so it should be available and populated for every simulation run.

## Table Schema

| Column        | Data Type            | Description                                                                |
|:--------------|:---------------------|:---------------------------------------------------------------------------|
| fare_id       | INTEGER PRIMARY KEY  | Unique identifier for a given fare [^1]                                    |
| carrier       | VARCHAR(10) NOT NULL | Name of carrier for this leg                                               |
| orig          | VARCHAR(10) NOT NULL | Origin (typically an airport code or similar)                              |
| dest          | VARCHAR(10) NOT NULL | Destination (typically an airport code or similar)                         |
| booking_class | VARCHAR(10) NOT NULL |                                                                            |
| price         | INTEGER              |                                                                            |
| restrictions  | VARCHAR(20) NOT NULL | Comma delimited list of fare restrictions                                  |
| category      | VARCHAR(20)          | Optional fare category (e.g. International, Domestic Restricted, etc) [^2] |

[^1]:
    `fare_id` values are not specified in the user's configuration file, but instead
    unique values are generated as Fare objects are created as the simulation is
    initialized.

[^2]:
    The `category` has no material effect on the simulation, but is stored in the
    database table and can be used for analysis of results after a simulation is
    complete, e.g. to look at results for just international bookings, or just
    in domestic restricted markets where LCC's are not operating, etc.
