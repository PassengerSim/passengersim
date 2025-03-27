# Fare Detail

The `fare_detail` database table stores details about the results of the
simulation at the fare level.

The table is created by the [`create_table_fare_detail`]
[passengersim.database.tables.create_table_fare_detail] function, and (potentially)
populated during a simulation run.  To be populated with data, one of the follow
flags must be set on [`Config.db.write_items`][passengersim.config.DatabaseConfig.write_items]:

- "fare": The table will be populated at each DCP during the simulation.

- "fare_final": The table will be populated only at the end of each sample (i.e. DCP 0) during the simulation

## Table Schema

| Column        | Data Type            | Description                                          |
|:--------------|:---------------------|:-----------------------------------------------------|
| scenario      | VARCHAR(20) NOT NULL | Scenario name [^1]                                   |
| iteration     | INT NOT NULL         |                                                      |
| trial         | INT NOT NULL         |                                                      |
| sample  	     | INT NOT NULL         | Sample number within trial                           |
| days_prior    | INT NOT NULL         | Days before departure                                |
| carrier       | VARCHAR(10) NOT NULL | Carrier name                                         |
| orig          | VARCHAR(10) NOT NULL | Origin (typically an airport code or similar)        |
| dest          | VARCHAR(10) NOT NULL | Destination (typically an airport code or similar)   |
| booking_class | VARCHAR(10) NOT NULL |                                                      |
| sold	         | INT                  | Number of customers buying this fare product         |
| sold_business | INT                  | Number of businss customers buying this fare product |
| price         | FLOAT                | Price of this fare                                   |
| updated_at    | DATETIME NOT NULL    | Time each row was written to the database            |


[^1]:
    The scenario name should be a string, and a unique name should be used for
    each unique scenario, which allows multiple scenarios to be saved in the
    same database.  When using SQLite (the default database engine) it is preferred
    to simply create a new database file for each unique scenario, but this
    database schema is designed to accommodate other database engines where that
    may be inconvenient.
