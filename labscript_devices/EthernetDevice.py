from blacs.tab_base_classes import Worker
import socket
import time

########################################################
# Communication config for Siglent SSG_3000X via Socket
# Port is set to 5025 by default
########################################################



class SocketWorker(Worker):

    def init(self):

        self.socket = None
        self.Host = self.IP_address.split("::")[0]
        self.Port = 5025#self.IP_address.split("::")[1]

        self.init_Connection()

    def init_Connection(self):

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.Host,self.Port))


    def shutdown(self):
        self.socket.close()
        time.sleep(.300)


        if self.rm is not None:
            self.rm.close()  # close ressource manager session
            self.rm = None

    def abort_transition_to_buffered(self):
        return self.transition_to_manual()

    def abort_buffered(self):
        return self.transition_to_manual()

    def transition_to_manual(self):
        return True  # return success
