from blacs.tab_base_classes import Worker


class GPIBWorker(Worker):

    def init(self):
        # Do not use visa anymore, as it is depricated (April 2021;LP)
        global visa
        import pyvisa as visa
        global h5py
        import labscript_utils.h5_lock
        import h5py

        self.rm = None
        self.GPIB_connection = None

        self.init_GPIB()

    def init_GPIB(self):
        # initialize the GPIB connection
        self.rm = visa.ResourceManager()
        # connect to GPIB device using the correct resource name
        self.GPIB_connection = self.rm.open_resource(self.GPIB_address)

        self.GPIB_connection.lock(requested_key=self.GPIB_address)
        #self.GPIB_connection.lock_excl()

    def shutdown(self):
        if self.GPIB_connection is not None:
            self.GPIB_connection.unlock()
            self.GPIB_connection.close()  # close GPIB connection
            self.GPIB_connection = None

        if self.rm is not None:
            self.rm.close()  # close ressource manager session
            self.rm = None

    def abort_transition_to_buffered(self):
        return self.transition_to_manual()

    def abort_buffered(self):
        return self.transition_to_manual()

    def transition_to_manual(self):
        return True  # return success
