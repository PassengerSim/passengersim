# Leg Summary

The `leg_summary` database table stores aggregate summary information
about the various legs in the simulation.  It is always generated after a
simulation run and does not need to be activated by any configuration setting.

## Table Schema

| Column    | Data Type | Description                                |
|:----------|:----------|:-------------------------------------------|
| carrier   | TEXT      | Carrier name                               |
| flt_no    | INTEGER   | Leg identifier                             |
| orig      | TEXT      | Leg Origin                                 |
| dest      | TEXT      | Leg Destination                            |
| avg_sold  | REAL      | Average number of seats sold               |
| avg_rev   | INTEGER   | Average revenue                            |
| lf        | REAL      | Average leg load factor                    |
