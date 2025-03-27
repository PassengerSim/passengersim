# Bookings by Timeframe

The `bookings_by_timeframe` database table stores aggregate summary information
about simulated bookings by timeframe, carrier, booking class, and passenger segment.
Unlike various other "detail" tables, it does not store the results from any single
sample, but instead has information aggregated over all samples in each trial.

The table is created by the [`create_table_bookings_by_timeframe`]
[passengersim.database.tables.create_table_bookings_by_timeframe] function, and (potentially)
populated during a simulation run.  To be populated with data, the "bookings"
flag must be set on [`Config.db.write_items`][passengersim.config.DatabaseConfig.write_items].

## Table Schema

| Column        | Data Type            | Description                                                           |
|:--------------|:---------------------|:----------------------------------------------------------------------|
| scenario      | VARCHAR(20) NOT NULL | Scenario name [^1]                                                    |
| trial         | INT NOT NULL         |                                                                       |
| carrier       | VARCHAR(10) NOT NULL | Carrier name                                                          |
| booking_class | VARCHAR(10) NOT NULL |                                                                       |
| days_prior    | INT NOT NULL         | Days before departure                                                 |
| tot_sold      | FLOAT                | Total sales for the carrier and booking class, through this timeframe |
| avg_sold      | FLOAT                | Average number of sales (per sample)                                  |
| avg_business  | FLOAT                | Average number of sales to business customers                         |
| avg_leisure   | FLOAT                | Average number of sales to leisure customers                          |
| avg_revenue   | FLOAT                | Average revenue from sales                                            |
| avg_price     | FLOAT                | Average price sold                                                    |
| updated_at    | DATETIME NOT NULL    | Time each row was written to the database                             |

[^1]:
    The scenario name should be a string, and a unique name should be used for
    each unique scenario, which allows multiple scenarios to be saved in the
    same database.  When using SQLite (the default database engine) it is preferred
    to simply create a new database file for each unique scenario, but this
    database schema is designed to accommodate other database engines where that
    may be inconvenient.
