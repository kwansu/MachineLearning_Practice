import numpy as np

def numerical_derivative(expression, x):
    limitDistance = 0.0001
    result = np.zeros_like(x)
    iterator = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

    while not iterator.finished:
        iterIndex = iterator.multi_index
        tempCopy = x[iterIndex]
        x[iterIndex] = tempCopy + limitDistance
        plusDx = expression(x)
        x[iterIndex] = tempCopy - limitDistance
        minusDx = expression(x)
        result[iterIndex] = (plusDx - minusDx) / (2 * limitDistance)
        x[iterIndex] = tempCopy
        iterator.iternext()
        
    return result
