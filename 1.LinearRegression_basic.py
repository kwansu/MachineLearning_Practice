from NumericalDifferentiation import differentiate
import numpy as np

x_data = np.array((1, 2, 3, 4, 5))
y_data = np.array((3, 1, -1, -3, -5))


def hypothesis(x, w, b):
    return x*w + b


def activateMeanSquaredError(y):
    return sum((y_data - y)**2)


def cost(x, w, b):
    return activateMeanSquaredError(hypothesis(x, w, b))


w = np.random.random(1)
b = np.random.random(1)

for i in range(1001):
    if i % 100 == 0:
        print(f'ephoc : {i}, cost : {cost(x_data, w, b)}')
    w -= 0.01 * differentiate(lambda t: cost(x_data, t, b), w)
    b -= 0.01 * differentiate(lambda t: cost(x_data, w, t), b)

print(f"w : {w}, b : {b}")
print(f"x : 4, predict : {hypothesis(4, w, b)}")
