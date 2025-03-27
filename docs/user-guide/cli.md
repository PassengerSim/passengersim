# Command Line Interface

AirSim Command Line Interface

**Usage**:

```console
$ python -m AirSim [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `info`
* `run`

## `python -m AirSim info`

**Usage**:

```console
$ python -m AirSim info [OPTIONS]
```

**Options**:

* `--help`: Show this message and exit.

## `python -m AirSim run`

**Usage**:

```console
$ python -m AirSim run [OPTIONS]
```

**Options**:

* `-n, --network-file PATH`: A file that defines the network and various simulation options.  [required]
* `-a, --airports-file PATH`: A file that defines the airports used in the simulation.
* `-o, --out-dir PATH`: Out files are written to this directory.
* `--db-engine TEXT`: [default: sqlite]
* `--db-filename TEXT`: Use this filename for the output database file. Applies to the the SQLite engine only.  [default: airsim-output.sqlite]
* `--fast / --slow`: For the SQLite engine only, running in 'fast' mode will store everything in an in-memory first, and dump the entire database to disk only when the simulation is complete.  This can be quite advantageous when the write speed of the disk is slow.  [default: slow]
* `--help`: Show this message and exit.
