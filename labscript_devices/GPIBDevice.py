from blacs.tab_base_classes import Worker
import zmq

class GPIBWorker(Worker):

    def init(self):
        # Do not use visa anymore, as it is depricated (April 2021;LP)
        global visa
        # import visa  # the NI-GPIB library
        # global pyvisa
        import pyvisa as visa
        global h5py
        import labscript_utils.h5_lock
        import h5py

        self.rm = None
        self.GPIB_connection = None
        self.run_thread = None

        self.context = zmq.Context()
        self.from_master_socket = self.context.socket(zmq.SUB)
        self.to_master_socket = self.context.socket(zmq.PUB)

        self.from_master_socket.connect(f"tcp://{self.jump_address}:44555")
        self.to_master_socket.connect(f"tcp://{self.jump_address}:44556")

        self.from_master_socket.subscribe("")

        self.init_GPIB()

    def init_GPIB(self):
        # initialize the GPIB connection
        self.rm = visa.ResourceManager()
        # connect to GPIB device using the correct resource name
        self.GPIB_connection = self.rm.open_resource(self.GPIB_address)

        self.GPIB_connection.lock(requested_key=self.GPIB_address)
        #self.GPIB_connection.lock_excl()

    def shutdown(self):

        self.from_master_socket.close()
        self.to_master_socket.close()

        self.context.term()

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
        if self.run_thread is not None:
            self.run_thread.join()
        return True  # return success
