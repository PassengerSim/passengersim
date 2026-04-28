# RM Action Base Classes

::: passengersim.rm.systems
    options:
        heading: "rm.systems"
        toc_label: "rm.systems"
        show_source: false
        show_symbol_type_toc: true
        show_symbol_type_heading: true
        show_root_full_path: true
        show_attribute_values: true
        filters: ["!^_[^_]", "!REGISTERED_SYSTEMS", "!__call__", "!__repr__"]
        members_order: source
        members:
          - RmAction
          - RmSys
          - RmSysOption
          - register_rm_system
          - get_registered_rm_system
          - check_registered_rm_system
          - list_registered_rm_systems
          - export_registered_rm_systems
          - restore_registered_rm_systems
        docstring_options:
            ignore_init_summary: false
            trim_doctest_flags: true
        merge_init_into_class: true
