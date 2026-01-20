from UKF.context import Context
from UKF.plotter import Plotter
from UKF.data_processor import DataProcessor
from pathlib import Path
import numpy as np

def chol(orig):
    ret = np.zeros(orig.shape)
    for i in range(4):
        ret[i, i] = orig[i, i]
        for k in range(i):
            ret[i, i] -= ret[i, k]*ret[i, k]
        if ret[i][i] <= 0:
            print("non pos def")
            return
        ret[i, i] = np.sqrt(ret[i, i])
        for j in range(i + 1, 4):
            ret[j, i] = orig[j, i]
            for k in range(i):
                ret[j, i] -= ret[j, k] * ret[i, k]
            ret[j, i] /= ret[i, i]
    return ret


#a = np.array([[2,1,0,0],[1,3,0,0],[0,0,1,0], [0,0,0,4]])
s = np.array([[5,2,0,0],[2,5,1,0],[0,1,3,1], [0,0,1,4]])
pxy = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16],[17,18,19,20]])
# print(pxy @ np.linalg.inv(s))
print(s)
print()
l = np.linalg.cholesky(s)
y = np.linalg.solve(l, pxy.T)
#print(np.linalg.solve(l.T, y))
print(l)
print()
print(chol(s))


# print("\n")

# residual = np.array([10,20,30,40])
# print(residual * (np.linalg.inv(s) @ residual))
# y = np.linalg.solve(l, residual.T)
# x = np.linalg.solve(l.T, y).T
# print(residual * x)


