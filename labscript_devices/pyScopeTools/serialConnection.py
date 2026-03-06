from __future__ import print_function
import serial
import io
import time

import logging, logging.handlers

logger = logging.getLogger('ScopeLogger')        
#handler = logging.handlers.RotatingFileHandler('C:/Temp/pyScopeTools.log', maxBytes=1024*1024*50)
#handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s'))
#handler.setLevel(logging.DEBUG)
#logger.setLevel(logging.DEBUG)
#logger.addHandler(handler) 

class SerialComm:

    def __init__(self, address, baudrate=9600, timeout=2000, eol='\r\n'):
        self.serial = serial.Serial(port=address, baudrate=baudrate)
        self.serial.flushInput()
        self.serial.flushOutput()
        self.serial.timeout = timeout
        self.eol = eol
        self.b_eol = eol.encode('ascii')
        
        self.do_init_oszi()
        
    def do_init_oszi(self):
        self.writeline("*CLS") #clear status
        self.writeline("RS232:TRANS:TERM CRLF") #config OSZI line termination characters

    def inWaiting(self):
        return self.serial.inWaiting()   
        
    def readline(self):
        logger.info("readline")
        out = b""
        sleep_amount = 0
        while (not self.b_eol in out) and sleep_amount < 5:
            if self.serial.inWaiting() > 0:
                out += self.serial.read(self.serial.inWaiting())
                logger.info(out.decode('ascii'))
                #time.sleep(0.01)
                sleep_amount = 0
            else:
                sleep_amount += 1
                time.sleep(0.1)

        if sleep_amount >= 5:
            print("Sleep amount limit is reached. so the message may not be complete")    
            logger.info('sleep amount reached')

        try:
            out = out.decode('ascii')
            logger.info("read: "+out)  
        except:
            print("Error parsing output: ")
            print(out)
        return out.replace(self.eol,'\n')

        
    def read(self, bytes=1):
        return self.serial.read(bytes).decode('ascii')
        
    def readline_raw(self, min_bytes=0):
        out = b""
        sleep_amount = 0
        while ((not self.b_eol in out) or len(out) < min_bytes) and sleep_amount < 5 : #min_bytes fixes a bug, when the data bytes contain a \r\n sequence.
            if self.serial.inWaiting() > 0:
                out += self.serial.read(self.serial.inWaiting())
                sleep_amount = 0
            else:
                sleep_amount += 1
                time.sleep(0.1)

        if sleep_amount >= 5:
            print("Sleep amount limit is reached. so the message may not be complete")    
            logger.info('sleep amount reached')

        return out.replace(self.b_eol,b'\n')

    def write(self, message):
        self.writeline(message)
    
    def writeline(self, message):
        logger.info("writeline: "+message)
        self.serial.write((message+self.eol).encode('ascii'))
        logger.info("in waiting: "+str(self.inWaiting()))
        
    def query(self, msg, timeout=None):
        logger.info("query")
        oldTimeout = self.serial.timeout
        if timeout:
            self.serial.timeout = timeout
        self.write(msg)
        result = self.readline()
        self.serial.timeout = oldTimeout
        return result