from NumericalDifferentiation import*

x_data = np.array([60, 73, 90, 55, 81, 70, 69, 95, 88, 83, 65])
y_data = [False, False, True, False, True,
          False, False, True, True, True, False]

x_data_normalization = (x_data - np.mean(x_data, axis=0)
                        ) / np.std(x_data, axis=0)
x_data_normalization = np.reshape(
    x_data_normalization, [len(x_data_normalization), 1])
y_data = np.array([(1. if element else 0.) for element in y_data])
y_data = np.reshape(y_data, [len(y_data), 1])


def hypothesisFunction(x, w, b):
    g = np.dot(x, w) + b
    temp = sigmoidFunction(g)
    return temp


def costFunction(x, w, b):
    h = hypothesisFunction(x, w, b)
    delta = 1e-7
    temp = y_data*np.log(h+delta) + (1-y_data)*np.log(1-h+delta)
    return -np.sum(temp)


W = np.array([1.])
W = np.reshape(W, [1, 1])
b = np.array([1.])


def predict(x):
    y = hypothesisFunction(x, W, b)
    iterator = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    _x = np.reshape(x_data, [11, 1])
    sameCount = 0

    while not iterator.finished:
        iterIndex = iterator.multi_index
        predic_value = True if y[iterIndex] >= 0.5 else False
        print("x : {} , predict : {}".format(_x[iterIndex], predic_value))
        if predic_value == bool(y_data[iterIndex]):
            sameCount+=1
        iterator.iternext()

    print("accuracy : %f" % (sameCount / y_data.size))


for i in range(1000):
    print('epoch %d, cost : %f' %
          (i, costFunction(x_data_normalization, W, b)))
    t1 = numerical_derivative(
        lambda t: costFunction(x_data_normalization, t, b), W)
    W -= 0.01 * t1
    b -= 0.01 * \
        numerical_derivative(lambda t: costFunction(
            x_data_normalization, W, t), b)

print("W : {}, b : {}".format(W, b))
predict(x_data_normalization)
