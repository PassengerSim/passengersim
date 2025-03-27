# Leg Detail

The `leg_detail` database table stores details about the results of the
simulation at the leg level.  Facts about the leg which are not dependent
on the simulation are stored in the [`leg_defs`](leg_defs.md) table instead.

The table is created by the [`create_table_leg_detail`]
[passengersim.database.tables.create_table_leg_detail] function, and (potentially)
populated during a simulation run.  To be populated with data, one of the follow
flags must be set on [`Config.db.write_items`][passengersim.config.DatabaseConfig.write_items]:

- "leg": The table will be populated at each DCP during the simulation.

- "leg_final": The table will be populated only at the end of each sample (i.e. DCP 0) during the simulation

- "leg_daily": The table will be populated at the end of each day during the simulation. Note
this will produce **a lot** of output, and is probably not desirable for most
simulation exercises.


## Table Schema

| Column             | Data Type            | Description                                       |
|:-------------------|:---------------------|:--------------------------------------------------|
| scenario           | VARCHAR(20) NOT NULL | Scenario name [^1]                                |
| iteration          | INT NOT NULL         |                                                   |
| trial              | INT NOT NULL         |                                                   |
| sample  	          | INT NOT NULL         | Sample number within trial                        |
| days_prior         | INT NOT NULL         | Days before departure                             |
| flt_no             | INT NOT NULL         | Unique identifier for a given leg [^3]            |
| updated_at         | DATETIME NOT NULL    | Time each row was written to the database         |
| sold	              | INT                  | Number of seats sold at this point in time        |
| revenue            | FLOAT                | Revenue attributed to this leg from seats sold    |
| q_demand           | FLOAT                |                                                   |
| untruncated_demand | FLOAT                |                                                   |
| forecast_mean      | FLOAT                | Forecast of mean demand to come before departure  |
| bid_price          | FLOAT                | Computed bid price for this leg at this time [^2] |

[^1]:
    The scenario name should be a string, and a unique name should be used for
    each unique scenario, which allows multiple scenarios to be saved in the
    same database.  When using SQLite (the default database engine) it is preferred
    to simply create a new database file for each unique scenario, but this
    database schema is designed to accommodate other database engines where that
    may be inconvenient.

[^2]:
    Bid prices are only computed if there is a RM that provides an instruction to
    do the computation.  If there is no calculated bid price, this column will be
    blank.

[^3]:
    In the "real world" the limitations of current technology make it such that
    flight numbers are not necessary unique by leg, as a single carrier may have
    multiple segments sharing the same flight number, and multiple carriers will
    have completely unrelated flights with the same flight number.  To simplify
    data processing, PassengerSim uses a unique id for every travel segment. Networks
    in PassengerSim that are derived from realistic sources will require some
    modest preprocessing to create unique flight numbers for every leg.
