from NumericalDifferentiation import differentiate
import numpy as np

x_data = np.array((1, 2, 3, 4, 5))
y_data = np.array((3, 1, -1, -3, -5))


def hypothesis(x, w, b):
    return x*w + b


def activate_meanSquaredError(p, y):
    return sum((y - p)**2)


def calculate_loss(x, y, w, b):
    return activate_meanSquaredError(hypothesis(x, w, b), y)


w = np.random.random(1)
b = np.random.random(1)

for i in range(1001):
    if i % 100 == 0:
        print(f'ephoc : {i}, loss : {calculate_loss(x_data, y_data, w, b)}')
    w -= 0.01 * differentiate(lambda t: calculate_loss(x_data, y_data, t, b), w)
    b -= 0.01 * differentiate(lambda t: calculate_loss(x_data, y_data, w, t), b)

print(f"w : {w}, b : {b}")
print(f"x : 4, predict : {hypothesis(4, w, b)}")
