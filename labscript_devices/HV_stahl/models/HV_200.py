from labscript_devices.HV_stahl.labscript_devices import HV_

CAPABILITIES = {
    'AO_range': 200,
    'num_AO': 8, # TODO: ?
    'baud_rate': 9600,
}

class HV_200(HV_):
    description = 'HV_200'

    def __init__(self, *args, **kwargs):
        """Class for HV 200"""
        combined_kwargs = CAPABILITIES.copy()
        combined_kwargs.update(kwargs)
        HV_.__init__(self, *args, **kwargs)