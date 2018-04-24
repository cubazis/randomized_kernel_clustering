import numpy as np
import src.nystrom as nys

X = np.array([[1, 2], [3, 4]])
print(nys.decom(X, nys.findRep(X, 2), 1, 'Poly', [2, 0]))

print(nys.nextPowerOf2(5))
print(nys.nextPowerOf2(17))
print(nys.nextPowerOf2(32))