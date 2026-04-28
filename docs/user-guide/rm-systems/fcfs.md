# Specialty Systems

PassengerSim includes several specialty revenue management systems that
implement specific algorithms or approaches.  In contrast to the standard
systems that are reflective of common RM systems deployed in "real world"
applications, the specialty systems are often more academic in nature, and
represent naive, historical, or otherwise non-optimal approaches to revenue
management.  Just because these systems are not optimal does not mean they are
not useful; they can be very helpful for software testing, benchmarking and
comparison purposes.

Unlike the standard RM systems, users must explicitly import specialty systems
if they wish to use them. This helps reduce the chance of accidental usage.


## First Come, First Served (FCFS)

::: passengersim.rm.specialty_systems.fcfs.FirstComeFirstServed
    options:
      show_root_heading: false
      show_root_full_path: false
      show_source: false


## No Detruncation

::: passengersim.rm.specialty_systems.e_no_detruncation.E_NoDetruncation
    options:
      show_root_heading: false
      show_root_full_path: false
      show_source: false
