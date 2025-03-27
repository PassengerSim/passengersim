# Leg Bucket Detail

The `leg_bucket_detail` database table stores details about the results of the
simulation at the leg-bucket level.

The table is created by the [`create_table_leg_bucket_detail`]
[passengersim.database.tables.create_table_leg_bucket_detail] function, and (potentially)
populated during a simulation run.  To be populated with data, one of the follow
flags must be set on [`Config.db.write_items`][passengersim.config.DatabaseConfig.write_items]:

- "bucket": The table will be populated at each DCP during the simulation.

- "bucket_final": The table will be populated only at the end of each sample (i.e. DCP 0) during the simulation

## Table Schema

| Column                    | Data Type            | Description                                                     |
|:--------------------------|:---------------------|:----------------------------------------------------------------|
| scenario                  | VARCHAR(20) NOT NULL | Scenario name [^1]                                              |
| iteration                 | INT NOT NULL         |                                                                 |
| trial                     | INT NOT NULL         |                                                                 |
| sample  	                 | INT NOT NULL         | Sample number within trial                                      |
| days_prior                | INT NOT NULL         | Days before departure                                           |
| flt_no                    | INT NOT NULL         | Unique identifier for a given leg [^3]                          |
| bucket_number             | INT NOT NULL         | Bucket number (sequential from 0)                               |
| name                      | VARCHAR(10) NOT NULL | Bucket name                                                     |
| auth	                     | INT                  | Number of seats in this bucket (or higher) available to be sold |
| revenue                   | FLOAT                | Revenue attributed to this bucket from seats sold               |
| sold	                     | INT                  | Number of seats sold at this point in time                      |
| untruncated_demand        | FLOAT                |                                                                 |
| forecast_mean             | FLOAT                |                                                                 |
| forecast_stdev            | FLOAT                |                                                                 |
| forecast_closed_in_tf     | FLOAT                |                                                                 |
| forecast_closed_in_future | FLOAT                |                                                                 |
| updated_at                | DATETIME NOT NULL    | Time each row was written to the database                       |


[^1]:
    The scenario name should be a string, and a unique name should be used for
    each unique scenario, which allows multiple scenarios to be saved in the
    same database.  When using SQLite (the default database engine) it is preferred
    to simply create a new database file for each unique scenario, but this
    database schema is designed to accommodate other database engines where that
    may be inconvenient.

[^3]:
    In the "real world" the limitations of current technology make it such that
    flight numbers are not necessary unique by leg, as a single carrier may have
    multiple segments sharing the same flight number, and multiple carriers will
    have completely unrelated flights with the same flight number.  To simplify
    data processing, PassengerSim uses a unique id for every travel segment. Networks
    in PassengerSim that are derived from realistic sources will require some
    modest preprocessing to create unique flight numbers for every leg.
