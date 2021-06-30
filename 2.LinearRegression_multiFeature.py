from NumericalDifferentiation import differentiate
import numpy as np

loadedData = np.loadtxt('data/linear_multiFeature.csv',
                        delimiter=',', dtype=np.float32)
x_data = loadedData[:, 0:-1]
y_data = loadedData[:, [-1]]


def hypothesis(x, W, b):
    return np.dot(x, W) + b


def activate_meanSquaredError(p, y):
    return sum((y - p)**2)


def predict(x):
    y = hypothesis(x, W, b)
    print("predict {} : {}".format(x, y))


def calculate_cost(x, y, w, b):
    return activate_meanSquaredError(hypothesis(x, w, b), y)


W = np.random.random((3, 1))
b = np.random.random(1)

for i in range(10001):
    if i % 100 == 0:
        print(f'ephoc : {i}, cost : {calculate_cost(x_data, y_data, W, b)}')
    W -= 0.000001 * differentiate(lambda t: calculate_cost(x_data, y_data, t, b), W)
    b -= 0.000001 * differentiate(lambda t: calculate_cost(x_data, y_data, W, t), b)

print("W : {}, b : {}".format(W, b))
print("x : (90,90,90), predict : %f" % hypothesis((90, 90, 90), W, b))
