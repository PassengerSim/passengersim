# Tutorials

This section includes some tutorials to demonstrate some of the capabilities of
PassengerSim.  These examples are built upon a very simple three market network
with two carriers.  Each airline offers a pair of flights from BOS to ORD, and
another pair from ORD to LAX.  Passenger demands are constructed for each single-leg
market, as well as for the connecting market BOS-LAX.

## 3MKT AL1

```mermaid
graph LR
    BOS-->|101| ORD;
    BOS-->|102| ORD;
    ORD-->|111| LAX;
    ORD-->|112| LAX;
```

## 3MKT AL2

```mermaid
graph LR
    BOS-->|201| ORD;
    BOS-->|202| ORD;
    ORD-->|211| LAX;
    ORD-->|212| LAX;
```
