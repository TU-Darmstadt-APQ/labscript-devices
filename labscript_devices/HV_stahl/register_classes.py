from labscript_devices import register_classes

register_classes(
    "HV_stahl",
    BLACS_tab='labscript_devices.HV_stahl.BLACS_tabs.HV_Tab',
    runviewer_parser=None,
)