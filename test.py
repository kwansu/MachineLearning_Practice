import numpy as np
from time import time

a = np.zeros((128,1000,1000))
a.fill(1)

start_time = time()
for _ in range(100):
    np.sum(a, axis=(-1,-2))
print(time() - start_time)


start_time = time()
for _ in range(100):
    np.sum(a, axis=(-2,-1))
print(time() - start_time)