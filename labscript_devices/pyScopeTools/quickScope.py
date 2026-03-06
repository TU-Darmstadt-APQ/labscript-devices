"""
Small tool to quick download the current oszi graph and save it as png- & txt-file

usage:
python quickScope.py COM1
"""

import scope
import sys
import numpy as np
import matplotlib.pyplot as plt
import time

args = sys.argv[1:]
port = "COM1"
if len(args)>0:
    port = args[0]
print("Reading scope using: "+port)

sc = scope.Scope(port, baudrate=9600)
time.sleep(0.5)

#x, y1, y2 = sc.readScope("CH1Ch2")
result = sc.readScope("CH1CH2")
x = result[0]
y1 = result[1]
y2 = None
if len(result)==3 and result[2] is not None:
    y2 = result[2]
    
dataname = time.strftime("%x").replace('/','-')+"_"+time.strftime("%H-%M-%S")

len_y1 = len(y1)
data1 = np.vstack((x[0:len_y1],y1)).T
np.savetxt("./"+dataname+"_CH1.txt", data1)
if y2 is not None:
    len_y2 = len(y2)
    data2 = np.vstack((x[0:len_y2],y2)).T
    np.savetxt("./"+dataname+"_CH2.txt", data2)

plt.plot(x[0:len_y1], y1, "r-")
if y2 is not None:
    plt.plot(x[0:len_y2], y2, 'b-')
plt.savefig("./"+dataname+".png", dpi=600)
#plt.show()