import numpy as np
import matplotlib.pyplot as plt
import scope

myscope=scope.Scope("COM7",baudrate=9600,debug=False)
#myscope=scope.Scope(u'GPIB0::1::INSTR',debug=True)

#reading 1 channel
#x, data1 = myscope.readScope("CH1")
#reading 2 channels
x,data1, data2=myscope.readScope("CH1CH2")

plt.plot(x,data1, "r-")
plt.plot(x,data2, "g-")

plt.show()