import numpy as np

def differentiate(expression, x):
    result = np.zeros_like(x)
    iterator = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

    while not iterator.finished:
        iterIndex = iterator.multi_index
        tempCopy = x[iterIndex]
        limitDistance = 1e-4 * tempCopy

        x[iterIndex] = float(tempCopy) + limitDistance
        plusDx = expression(x)

        x[iterIndex] = float(tempCopy) - limitDistance
        minusDx = expression(x)

        result[iterIndex] = (plusDx - minusDx) / (2 * limitDistance)
        x[iterIndex] = tempCopy
        iterator.iternext()
        
    return result
