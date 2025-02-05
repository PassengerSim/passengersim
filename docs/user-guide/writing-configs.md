# Config Files

Most control of the simulation is done via configuration files written in YAML
format.


## Simulation Controls

```yaml
scenario: Three Market Test Network
simulation_controls:
  random_seed: 42
  num_trials: 1
  num_samples: 300
  sys_k_factor: 0.1
  mkt_k_factor: 0.2
  pax_type_k_factor: 0.4
  tf_k_factor: 0.1
  tot_z_factor: 2.0
  tf_z_factor: 2.0
  prorate_revenue: true
  dwm_lite: false
  max_connect_time: 120
  disable_ap: false
  demand_multiplier: 1.0
  manual_paths: true
```

## RM Systems

The [`rm_systems`][passengersim.config.Config.rm_systems] key allows the user
to define one or more revenue management systems that may be used by carriers.


These systems can either be defined as a list, where each item in the list defines
both a name and steps, or you can write the same instruction as a nested mapping,
with the names as keys and the values giving the other attributes of each RM system,
(for now, just a list of steps) like this:


=== "as list"

    ```yaml
    rm_systems:
    - name: rm_test1
      steps:
      - step_type: untruncation  #(1)!
        name: untruncation
        algorithm: em
        kind: leg
      - step_type: forecast
        name: forecast
        algorithm: exp_smoothing
        alpha: 0.1
        kind: leg
      - step_type: optimization
        name: optimization
        algorithm: emsrb  #(2)!
        kind: leg
    ```
    { .annotate }

    1.  :book: [Untruncation](./untruncation.md) allows us to figure out how much demand was censored.
    2.  If you define different RM systems, you can attach different optimization algorithms, such as
        ProBP.


=== "as dict"

    ```yaml
    rm_systems:
      rm_test1:
        steps:
        - step_type: untruncation  #(1)!
          name: untruncation
          algorithm: em
          kind: leg
        - step_type: forecast
          name: forecast
          algorithm: exp_smoothing
          alpha: 0.1
          kind: leg
        - step_type: optimization
          name: optimization
          algorithm: emsrb  #(2)!
          kind: leg
    ```
    { .annotate }

    1.  :book: [Untruncation](./untruncation.md) allows us to figure out how much demand was censored.
    2.  If you define different RM systems, you can attach different optimization algorithms, such as
        ProBP.


## Passenger Choice Models

```yaml
choice_models:
  business:
    kind: pods
    emult: 1.6
    basefare_mult: 2.5
    path_quality:  [38.30,  0.10]
    airline_pref_pods:  [-12.29,  0.17]
    tolerance: 2.0
    r1: 0.30
    r2: 0.10
    r3: 0.20
    r4: 0.15
  leisure:
    kind: pods
    emult: 1.5
    basefare_mult: 1.0
    path_quality:  [2.02, 0.12]
    airline_pref_pods:  [-1.98, 0.11]
    tolerance: 5.0
    r1: 0.30
    r2: 0.15
    r3: 0.25
    r4: 0.20
```


## Define Carriers

```yaml
carriers:
- name: AL1
  rm_system: rm_test1
- name: AL2
  rm_system: rm_test1
- name: AL3
  rm_system: rm_test1
- name: AL4
  rm_system: rm_test1
```


## Define Booking Classes

```yaml
classes:
- Y0
- Y1
- Y2
- Y3
- Y4
- Y5
- Y6
- Y7
- Y8
- Y9
```


## Data Collection Points (DCPs)

```yaml
dcps:
- 63
- 56
- 49
- 42
- 35
- 31
- 28
- 24
- 21
- 17
- 14
- 10
- 7
- 5
- 3
- 1
```


## Booking Curves

```yaml
booking_curves:
- name: '1'
  curve:
    63: 0.01
    56: 0.02
    49: 0.05
    42: 0.13
    35: 0.19
    31: 0.23
    28: 0.29
    24: 0.35
    21: 0.45
    17: 0.54
    14: 0.67
    10: 0.79
    7: 0.86
    5: 0.91
    3: 0.96
    1: 1.0
- name: '2'
  curve:
    63: 0.13
    56: 0.22
    49: 0.37
    42: 0.52
    35: 0.64
    31: 0.7
    28: 0.75
    24: 0.78
    21: 0.83
    17: 0.87
    14: 0.91
    10: 0.94
    7: 0.96
    5: 0.98
    3: 0.99
    1: 1.0
- name: '3'
  curve:
    63: 0.04
    56: 0.06
    49: 0.12
    42: 0.26
    35: 0.35
    31: 0.41
    28: 0.48
    24: 0.54
    21: 0.63
    17: 0.7
    14: 0.81
    10: 0.88
    7: 0.93
    5: 0.96
    3: 0.98
    1: 1.0
- name: '4'
  curve:
    63: 0.21
    56: 0.35
    49: 0.53
    42: 0.67
    35: 0.76
    31: 0.8
    28: 0.83
    24: 0.85
    21: 0.88
    17: 0.91
    14: 0.94
    10: 0.96
    7: 0.97
    5: 0.98
    3: 0.99
    1: 1.0
```


## Legs

```yaml
legs:
- carrier: AL1
  fltno: 1
  orig: BOS
  dest: SFO
  date: '2020-01-01'
  dep_time: 08:00
  arr_time: '10:00'
  capacity: 100
  distance: 867.0
- carrier: AL2
  fltno: 2
  orig: BOS
  dest: SFO
  date: '2020-01-01'
  dep_time: '14:00'
  arr_time: '16:00'
  capacity: 100
  distance: 867.0
...
```


## Paths

```yaml

paths:
- orig: BOS
  dest: SFO
  path_quality_index: 1.0
  legs:
  - 1
- orig: BOS
  dest: SFO
  path_quality_index: 1.0
  legs:
  - 2
- orig: BOS
  dest: ORD
  path_quality_index: 1.0
  legs:
  - 3
...
```
