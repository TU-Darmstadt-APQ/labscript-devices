import labscript_devices 

labscript_devices.register_classes(
    'AgilentE4422B',
    BLACS_tab='labscript_devices.AgilentE4422B.blacs_tabs.AgilentE4422BTab',
    runviewer_parser=None
)