import labscript_devices

labscript_devices.register_classes(
    'GPIBLANAdapter',
    BLACS_tab='labscript_devices.GPIB_LAN_Adapter.blacs_tabs.GPIBLANAdapterTab',
    runviewer_parser=None
)
