from types import prepare_class
import numpy as np
from numpy.lib.function_base import gradient


class Layer:
    def __init__(self, input_shape, output_shape):
        self.output_shape = output_shape
        self.w = np.random.random((input_shape, output_shape))
        self.b = np.random.random((output_shape))
        self.g = None
        self.ig = None
        self.s = None
        # self.og = None

    
    def reset(self):
        self.w = np.random.random(self.w.shape)
        self.b = np.random.random(self.b.shape)


    def progress(self, x_input, input_gradient):
        self.ig = input_gradient
        self.g = np.sum(self.w, axis=(-2,-1))
        return self.activate_sigmoid(np.matmul(x_input, self.w)+ self.b)


    def activate_sigmoid(self, z):
        self.s = 1 / (1+np.exp(-z))
        return self.s


class Model:
    def __init__(self):
        self.layers = []


    def reset(self):
        for layer in self.layers:
            layer.reset()
    

    def add_layer(self, output_shape, input_shape = None):
        if input_shape is None:
            input_shape = self.layers[-1].output_shape
        self.layers.append(Layer(input_shape, output_shape))


    def predict(self, x):
        p = x
        gi = 1.0
        g = None
        for layer in self.layers:
            p = layer.progress(p, gi)
            g = layer.s * layer.g * layer.ig
            gi = g * (1-layer.s)
        return p, g


    # def binaryCrossentropy(self, p, y):
    #     delta = 0.0000001
    #     return -np.sum(y*np.log(p+delta) + (1-y)*np.log(1-p+delta))


    # def calc_loss(self, x, y):
    #     p, gradient = self.predict(x)
    #     return self.binaryCrossentropy(p, y)


    def update_layer(self, x, y):
        _, g = self.predict(x)
        g -= y



    def fit(self, x, y, epochs, learning_rate = 0.01, print_count = None):
        if print_count is None:
            print_count = epochs / 10
        for i in range(epochs):
            # if i%print_count == 0:
            #     print(f'ephoc : {i}, loss : {self.calc_loss(x, y)}')
            self.update_layer(x,y)

    
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
model.add_layer(input_shape=2, output_shape=3)
model.add_layer(1)


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
