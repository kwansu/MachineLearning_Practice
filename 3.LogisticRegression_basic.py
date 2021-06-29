from NumericalDifferentiation import*

x_data = np.array([60, 73, 90, 55, 81, 70, 69, 95, 88, 83, 65])
y_data = [False, False, True, False, True,
          False, False, True, True, True, False]

x_data_normalized = (x_data - np.mean(x_data, axis=0)) / np.std(x_data, axis=0)
x_data_normalized = np.reshape(x_data_normalized, [len(x_data_normalized), 1])
y_data = np.array([(1. if element else 0.) for element in y_data])
y_data = np.reshape(y_data, [len(y_data), 1])


def activateSigmoid(z):
    return 1 / (1+np.exp(-z))


def hypothesis(x, w, b):
    g = np.dot(x, w) + b
    return activateSigmoid(g)


def binaryCrossentropy(p):
    delta = 1e-7
    return -np.sum(y_data*np.log(p+delta) + (1-y_data)*np.log(1-p+delta))


w = np.random.random((1, 1))
b = np.random.random(1)
cost = lambda _x,_w,_b: binaryCrossentropy((hypothesis(_x,_w,_b)))

for i in range(10001):
    if i % 100 == 0:
        print('epoch %d, cost : %f' %(i, cost(x_data_normalized, w, b)))
    w -= (0.01 * differentiate(lambda t: cost(x_data_normalized, t, b), w))
    b -= (0.01 * differentiate(lambda t: cost(x_data_normalized, w, t), b))

print("w : {}, b : {}".format(w, b))


def predict(x):
    y = hypothesis(x, w, b)
    iterator = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    _x = np.reshape(x_data, [11, 1])
    correctCount = 0

    while not iterator.finished:
        iterIndex = iterator.multi_index
        predic_value = True if y[iterIndex] >= 0.5 else False
        print("x : {} , predict : {}".format(_x[iterIndex], predic_value))
        if predic_value == bool(y_data[iterIndex]):
            correctCount += 1
        iterator.iternext()

    print("accuracy : %f" % (correctCount / y_data.size))


predict(x_data_normalized)
