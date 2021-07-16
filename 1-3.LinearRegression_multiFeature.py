import numpy as np

loadedData = np.loadtxt('data/linear_multiFeature.csv',
                        delimiter=',', dtype=np.float32)
x_data = loadedData[:, 0:-1]
y_data = loadedData[:, [-1]]


def hypothesis(x, W, b):
    return np.dot(x, W) + b


def calc_loss(x, y, W, b):  # mean square error
    return np.sum((hypothesis(x,W,b) - y)**2)/len(y)


def differentiate(f, x):
    gradient = np.zeros_like(x)
    x_iter = np.nditer(x, flags=['multi_index'])

    while not x_iter.finished:
        mi = x_iter.multi_index
        source = x[mi]
        dx = 1e-4 * source
        y = f(x)
        x[mi] = source + dx
        y_plus_dx = f(x)
        gradient[mi] = (y_plus_dx - y) / dx
        x[mi] = source
        x_iter.iternext()
    return gradient


def train(x, y, w, b, *, learning_rate = 0.01, epochs = 1000, bins = 10):
    interval = int(epochs / bins)
    for i in range(epochs):
        if i % interval == 0:
            print(f'ephoc : {i}, loss : {calc_loss(x, y, w, b)}')
        w -= learning_rate * differentiate(lambda v: calc_loss(x, y, v, b), w)
        b -= learning_rate * differentiate(lambda v: calc_loss(x, y, w, v), b)
    print(f'ephoc : {i}, loss : {calc_loss(x, y, w, b)}')


W = np.random.random((3, 1))
b = np.random.random(1)

train(x_data, y_data, W, b, learning_rate=0.00001, epochs=10000)
print(f"W : {W}, b : {b}")
print(f"x : (90,90,90), predict : {hypothesis((90, 90, 90), W, b)}")
