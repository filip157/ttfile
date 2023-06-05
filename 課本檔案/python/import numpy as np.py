import numpy as np
import matplotlib.pyplot as plt


x = np.array([1,2,3,4,5])
y = x*2
plt.plot(x,y,"ro")
plt.axis([-10,10,-50,50])
plt.grid()
plt.show()