import numpy as np
a = np.array([1,2,3,4,5])
print(a)

a = np.array([[1,2],[1,2]])
b = np.array([[1,2],[1,2]])
print(a,"\n")
print(a[0,],"\n")
print(a[0,0],"\n")

c = a.dot(b)
print(c)