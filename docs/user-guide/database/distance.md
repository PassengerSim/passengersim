# Distance

The `distance` database table stores the distance between locations (airports
and/or other travel nodes). It contains static data not dependent on running
the simulation. The table is created by the [`create_table_distance`]
[passengersim.database.tables.create_table_distance] function.

## Table Schema

| Column        | Data Type            | Description                                          |
|:--------------|:---------------------|:-----------------------------------------------------|
| orig          | VARCHAR(10) NOT NULL | Origin (typically an airport code or similar)        |
| dest          | VARCHAR(10) NOT NULL | Destination (typically an airport code or similar)   |
| miles         | FLOAT                | Distance between nodes in miles                      |
