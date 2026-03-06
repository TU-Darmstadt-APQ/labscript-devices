from __future__ import print_function
import time
import numpy as np
import traceback

class Scope:
    """Object modeling the oszilloscope

    Working example:
#####
import numpy as np
import matplotlib.pyplot as plt
import scope

#myscope=scope.Scope(u'GPIB0::1::INSTR', debug=True)
myscope=scope.Scope("COM1", debug=False)

x,data=myscope.readScope(fast_mode=False)

plt.plot(x,data)
plt.show()
#####
    """

#using serial (COM) (@baudrate: 9600)
#   normal mode: 5.3sec
#   fast mode:   2.7sec
#using GPIB
#   normal mode: 0.55sec
#   fast mode:   0.37sec

    def __init__(self, address, baudrate=9600, timeout=5, debug = False):
        """Create the scope object with given parameters

        Arguments:
        address - the com port to be used, eg "COM1"
                  or a GPIB address LQ_GPIB like 'LQ_GPIB:1' (for using a LQElectronics UGPlus USB to GPIB Controller)
                  or a GPIB address like "u'GPIB0::1::INSTR'" (for using a Nation Instruments GPIB adapter)
                  the address is obtained by using GPIBConnection.list_devices()

        Keyword arguments:
        baudrate - serial baudrate that is used (default: 9600)
        timeout - set the timeout. NOTICE: a serial connection needs a long timeout, so the default value is 6sec!
        debug - switch the command line output
        """
        if "COM" in address:
            import serialConnection
            self.con = serialConnection.SerialComm(address, baudrate, timeout=timeout, eol='\r\n') #timeout in s
        elif "LQ_GPIB:" in address:
            import LQ_GPIBConnection
            self.con = LQ_GPIBConnection.LQ_GPIBComm(int(address[8:]))
        elif "GPIB" in address:
            import GPIBConnection
            self.con = GPIBConnection.GPIBComm(address, timeout=timeout*1000)#, eol='\r\n')          #timeout in ms
        else:
            print("No valid address type")
        self.debug = debug

        self.last_params = {}
        self.last_params['XZERO'] = 0.0

        #self.con.writeline("ACQ:STATE 1") #set oszi non freezing
        
    def get_oszi_ID(self):
        """
        Query the ID of the oszi.    
        """
        result = self.con.query("ID?")
        return result

    def unfreeze(self):
        self.con.writeline("ACQ:STATE 1")    
        
    def config_channel(self, channel=1, bandwidth=None,  coupling=None, position=None, scale=None, probe=None): #enable, scale, x-position?
        """
        Configure channel settings.

        This method configures the selected channel. Passing None to an argument will leave
        out part of the configuration and will not change it on the oszi.

        Parameters
        ----------
        channel : int [0, 1]
            This parameters selects the channel.
        bandwidth : str ["ON", "OFF"]
            This parameter specifies which bandwidth the oszi should use. "Off" means full (60MHz),
            "ON" means 20MHz
        coupling : str ["DC", "AC", "GND"]
            This parameter secifies which coupling should be used.
        position : float
            This parameter specifies the vertical position in Volts.
        scale : float
            This paramater specifies the vertical resolution. (height in Volts of one box on the oszi)
        probe : int [1, 10, 100, 1000]
            This parameter specifies the probe amount? (increases input resistance and higher
            measured voltages)

        Returns
        -------
        bool
            False if no configuration needed to be sent (because all arguments were None), otherwise True     
        """
        if channel not in (1, 2):
            raise InvalidArgumentException("The channel must be 1 or 2")
        if bandwidth and bandwidth not in ("ON", "OFF"):
            raise InvalidArgumentException("The bandwidth must be 'OFF' (full, 60MHz) or 'ON' (20MHz)") 
        if coupling and coupling not in ("DC", "AC", "GND"):
            raise InvalidArgumentException("The coupling must be 'DC', 'AC' or 'GND'")
        if (probe is not None) and probe not in (1, 10, 100, 1000):
            raise InvalidArgumentException("The probe must be 1, 10, 100 or 1000")    

        command = ":CH{:d}:".format(channel)
        if probe is not None:
            command += "PRO {:d};".format(probe)
        if scale is not None:
            command += "SCA {:.2E};".format(scale)
        if position is not None:
            command += "POS {:.2E};".format(position)
        if coupling:
            command += "COUP "+coupling+";"
        if bandwidth:
            command += "BAN "+bandwidth+";"
        command = command[:-1] #crop trailing ';'                    
        #self.con.writeline(":CH{:d}:PROBE {:d};SCALE {:.2E};POSITION {:.2E};COUPLING {:s};BANDWIDTH {:s}".format(channel, probe, scale, position, coupling, bandwidth))
        if len(command) > 6:
            self.con.writeline(command)
            return True
        return False    

    def get_channel_config(self, channel=1):
        result = self.con.query("CH"+str(channel)+"?")
        result = result[5:-1] #crop leading ":CH1:" and trailing "\n"
        split = result.split(';')
        result = {}
        for s in split:
            ss = s.split()
            result[ss[0]] = ss[1]    
        result['POSITION'] = eval(result['POSITION']) #convert to float
        result['SCALE'] = eval(result['SCALE'])
        result['PROBE'] = eval(result['PROBE'])    
        return result
    
    def get_time_config(self):
        result = self.con.query("HOR:MAI?")
        print("result = "+result)
        result = result[17:-1] #crop leading :HORIZONTAL:MAIN: & trailing \n
        print("result = "+result)
        split = result.split(";")
        print("split = "+str(split))
        result = {}
        for s in split:
            print("read: "+s)
            ss = s.split()
            result[ss[0]] = ss[1]
        result['SCALE'] = eval(result['SCALE'])
        result['POSITION'] = eval(result['POSITION'])

        return result

    def config_time(self, position=0.0, scale=1.0):
        command  = ":HOR:"
        if position is not None:
            command += "POS {:.2E};".format(position)
        if scale is not None:
            command += "SCA {:.2E};".format(scale)
        command = command[:-1] #crop trailing ';'        
        #self.con.writeline(":HOR:POS {:.2E};SCA {:.2E}".format(position, scale))
        if len(command) > 6:
            self.con.writeline(command)
            return True
        return False    
        
    def get_trigger_config(self):
        result = self.con.query(":TRIG:MAI?")
        result = result[:-1] #crop trailing \n
        split = result.split(";")
        result = {}
        for s in split:
            ss = s.split()
            if "MODE" in ss[0]:
                result['MODE'] = ss[1] #NORMAL, AUTO
            if "TYPE" in ss[0]:
                result['TYPE'] = ss[1] #EDGE, (VIDEO)
            if "COUPLING" in ss[0]:
                result['COUPLING'] = ss[1] #DC, NOISEREJ, HFREJ, LFREJ, AC
            if "SLOPE" in ss[0]:
                result['SLOPE'] = ss[1] #RISE, FALL
            if "LEVEL" in ss[0]:    
                result['LEVEL'] = ss[1]

        return result
    
    def config_trigger(self, mode=None, typ=None, coupling=None, slope=None, level=None):
        if mode and mode not in ("NORMAL", "AUTO"):
            raise InvalidArgumentException("Mode must be 'NORMAL' or 'AUTO'")
        if typ and typ not in ("EDGE", "VIDEO"):
            raise InvalidArgumentException("Type must be 'EDGE' or 'VIDEO'")
        if coupling and coupling not in ("AC", "DC", "NOISEREJ", "HFREJ", "NJREJ"):    
            raise InvalidArgumentException("coupling must be 'AC', 'DC', 'NOISEREJ', 'HFREJ' or 'NJREJ'")
        if slope and slope not in ("RISE", "FALL"):
            raise InvalidArgumentException("slope must be 'RISE' or 'FALL'")

        command = ""
        if mode or typ or (level is not None):
            command = ":TRIG:MAI:"
            if mode:
                command += "MOD "+mode+";"
            if typ:
                command += "TYP "+typ+";"
            if level is not None:
                command += "LEV {:.2E};".format(level)

        if coupling or slope:
            command += ":TRIG:MAI:EDGE:"
            if coupling:
                command += "COUP "+coupling+";"
            if slope:
                command += "SLO "+slope +";"
        command = command[:-1] #crop trailing ';'

        if len(command) > 11:
            self.con.writeline(command)
            return True
        return False    

    def readScope(self, channel="CH1", fast_mode=True, freeze_scope=False):
        """
        Read the data from scope without changing settings

        Keyword arguments:
        channel - channel to be read (default: "CH1"), also possible "CH1CH2"
        fast_mode - use 1byte vs 2bytes per data point
        freeze_scope - true, if the acquisition is stopped during transfering the data
        """
        #logger.info("now read scope")
        byte_wid = 1 if fast_mode else 2
        read_ch1 = "CH1" in channel
        read_ch2 = "CH2" in channel
        if not read_ch1 and not read_ch2:
            print("No channels selected.")
            return
        # catch possible exceptions
        try:
            t0 = time.time()
            self.con.writeline("HEAD ON")
            if freeze_scope:
                #freeze the Oszi
                if self.debug: print("freeze oszi")
                self.con.writeline("ACQ:STATE 0")
            #setup encoding
            if self.debug: print("setup encoding")
            self.con.writeline("DAT:ENC RIB")
            self.con.writeline("DAT:WID "+str(byte_wid))
            #setup start and endpoint
            if self.debug: print("setup start & end value")
            self.con.writeline("DAT:STAR 1")
            self.con.writeline("DAT:STOP 2500")
            #now read CH1
            if read_ch1:
                if self.debug: print("request CH1")
                self.con.writeline("DAT:SOU CH1")
                time.sleep(0.2)
                #out = self.con.query("WFMPRe:XINCR?;XZERO?;YMULT?;YZERO?;YOFF?") #only request neccessary parameters (not complete WFMPRe?)
                out = self.con.query("WFMPRe?")

                params = self.last_params
                params['XZERO'] = 0.0 #if the oszi does not support this, add a default value (for TDS 744A)
                out = out.split(';')
                for s in out:
                    d = s.split(" ") #now split on the space sign
                    #params[d[0][8:]] = d[1]
                    params[d[0]] = d[1]
                self.last_params = params

                self.con.writeline("CURV?") #request the curve data  
                out = self.con.readline_raw(2500 if byte_wid==1 else 5000)
                out = out[13:-1] #drop the first 13 chars: ":CURVE #45000" and the last '\n'
                #so now out contains only the bytes of the curve data
                
                if byte_wid == 1:
                    if len(out) > 2500: #should not happen
                        out = out[:2500]
                        print("1: must drop one")
                    data1 = np.fromstring(out, dtype='b')
                else:
                    if len(out) > 5000: #should not happen
                        out = out[:5000]
                        print("2: must drop one")
                    data1 = np.fromstring(out, dtype='>i2')
                    
                # add y-offset to data
                data1 = np.add(data1,np.ones(data1.shape)*(-float(params['YOFF'])))
                # multiply value to get the actual voltage
                data1 *= float(params['YMULT'])
                
                print("Len Data1: "+str(len(data1)))
            #now read CH2    
            if read_ch2:            
                if self.debug: print("request CH2")
                self.con.writeline("DAT:SOU CH2")
                time.sleep(0.2)
                #out = self.con.query("WFMPRe:XINCR?;XZERO?;YMULT?;YZERO?;YOFF?") #only request neccessary parameters (not complete WFMPRe?)
                #maybe also request XUNIT and YUNIT? but it seems to be always sec and Volts
                out = self.con.query("WFMPRe?")
                bkp_out = out
                params = self.last_params
                #params = {}
                params['XZERO'] = 0.0 #if the oszi does not support this, add a default value (for TDS 744A)
                
                out = out.split(';')
                for s in out:
                    d = s.split(" ") #now split on the space sign
                    #params[d[0][8:]] = d[1]
                    params[d[0]] = d[1]

                self.last_params = params 

                self.con.writeline("CURV?") #request the curve data
                out = self.con.readline_raw(2500 if byte_wid==1 else 5000)
                out = out[13:-1]                
                
                if byte_wid == 1:
                    if len(out) > 2500:
                        out = out[:2500]
                        print("1: must drop one")
                    data2 = np.fromstring(out, dtype='b')
                else:
                    if len(out) > 5000:
                        out = out[:5000] #drop another char  (only neccessary when using serial connection COM port)??
                        print("2: must drop one")
                    data2 = np.fromstring(out, dtype='>i2')
                    
                # add offset to data
                data2 = np.add(data2,np.ones(data2.shape)*(-float(params['YOFF'])))
                # multiply to get voltage
                data2 *= float(params['YMULT'])
                print("Len Data2: "+str(len(data2)))
    
            # create x axis
            y_amount = max(len(data1), len(data2)) #2500
            if not 'XINCR' in params:
                raise Exception("XINCR not in params. input str="+bkp_out)
            x = np.arange(float(params['XZERO']),
                            float(params['XZERO'])+y_amount*float(params['XINCR']),
                            float(params['XINCR']))    
                            
            
            if freeze_scope:
                #finally unfreeze the Oszi
                if self.debug: print("unfreeze oszi")
                self.con.writeline("ACQ:STATE 1")
            if self.debug: print("done")
            if self.debug:
                print("Reading took: "+str(time.time() - t0)+"sec")    
            if read_ch1 and read_ch2:
                return x, data1, data2
            if read_ch1:
                return x, data1
            if read_ch2:
                return x, data2
            else:
                return (0,0)
             
        except:
            # error in read
            traceback.print_exc()
            raise
            try:
                print("An Error occurred")
                self.con.close()
                return (0,0)
            except:
                print("Error Closing")
                return (0,0)