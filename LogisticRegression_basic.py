from NumericalDifferentiation import numerical_derivative
import numpy as np
import math

x_data = [60, 73, 90, 55, 81, 70, 69, 95, 88, 83, 65]
y_data = [False, False, True, False, True,
          False, False, False, True, True, True, False]

y_data = [(1 if element else 0) for element in y_data]

def hypothesisFuction(x,w,b):
    return np.dot(x,w) + b


def sigmoidFunction(f):
    return 1/(1-math.exp(f))


def costFunction(x,w,b):
   pass 

print(y_data)
