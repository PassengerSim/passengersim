# Carrier Summary

The `carrier_summary` database table stores aggregate summary information
about the various carriers in the simulation.  It is always generated after a
simulation run and does not need to be activated by any configuration setting.

## Table Schema

| Column       | Data Type | Description                                |
|:-------------|:----------|:-------------------------------------------|
| carrier      | TEXT      | Carrier name                               |
| sold         | REAL      | Average number of sold                     |
| sys_lf       | REAL      | Average system load factor                 |
| avg_leg_lf   | REAL      | Average leg load factor                    |
| avg_rev      | INTEGER   | Average revenue                            |
| avg_price    | REAL      | Average price of seats sold                |
| asm          | INTEGER   | Available seat miles                       |
| rpm          | INTEGER   | Revenue passenger miles                    |
| yield        | REAL      | Average revenue per revenue passenger mile |
