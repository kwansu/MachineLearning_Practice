from performance_test import print_execution_time
import numpy as np


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
    

class Layer:
    def __init__(self, input_shape, output_shape):
        self.output_shape = output_shape
        self.w = np.random.random((input_shape, output_shape))
        self.b = np.random.random((output_shape))

    
    def reset(self):
        self.w = np.random.random(self.w.shape)
        self.b = np.random.random(self.b.shape)


    def progress(self, x_input):
        temp = np.dot(x_input, self.w)
        return Layer.activate_sigmoid(temp+ self.b)


    def activate_sigmoid(z):
        return 1 / (1+np.exp(-z))


class Model:
    def __init__(self):
        self.layers = []


    def reset(self):
        for layer in self.layers:
            layer.reset()
    

    def add_layer(self, output_shape, input_shape=None):
        if input_shape is None:
            input_shape = self.layers[-1].output_shape
        self.layers.append(Layer(input_shape, output_shape))


    def predict(self, x):
        p = x
        for layer in self.layers:
            p = layer.progress(p)
        return p


    def binaryCrossentropy(self, p, y):
        delta = 0.0000001
        return -np.sum(y*np.log(p+delta) + (1-y)*np.log(1-p+delta))


    def calc_loss(self, x, y):
        p = self.predict(x)
        return self.binaryCrossentropy(p, y)


    def fit(self, x, y, epochs, learning_rate = 0.01, print_count = None):
        if print_count is None:
            print_count = epochs / 10
        for i in range(epochs):
            if i%print_count == 0:
                print(f'ephoc : {i}, loss : {self.calc_loss(x, y)}')
            for layer in self.layers:
                layer.w -= learning_rate * differentiate(lambda t: self.calc_loss(x, y), layer.w)
                layer.b -= learning_rate * differentiate(lambda t: self.calc_loss(x, y), layer.b)

    
    def evaluate(self, x, y):
        p = self.predict(x)
        correct_count = 0

        for xi, yi, pi in zip(x, y, p):
            y_predic = pi >= 0.5
            print(f"x : {xi} , predict : {y_predic}")
            if yi == y_predic:
                correct_count += 1

        print(f"accuracy : {correct_count/p.size}")


model = Model()
model.add_layer(input_shape=2, output_shape=4)
model.add_layer(1)


@print_execution_time
def test(x,y):
    model.reset()
    model.fit(x,y,epochs=10000)
    model.evaluate(x,y)


# or
x_data = np.array([(0, 0), (0, 1), (1, 0), (1, 1)])
y_data = np.array([0, 1, 1, 1]).reshape(4, 1)
test(x_data, y_data)

# and
x_data = np.array([(0, 0), (0, 1), (1, 0), (1, 1)])
y_data = np.array([0, 0, 0, 1]).reshape(4, 1)
test(x_data, y_data)

# xor
x_data = np.array([(0, 0), (0, 1), (1, 0), (1, 1)])
y_data = np.array([0, 1, 1, 0]).reshape(4, 1)
test(x_data, y_data)
