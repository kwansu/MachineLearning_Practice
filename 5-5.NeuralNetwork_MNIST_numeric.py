from numpy import random
from numpy.lib.utils import source
import pandas as pd
import numpy as np


class Layer:
    beta1 = 0.9
    beta2 = 0.999
    momentum_rate = 0.9
    epsilon = 1e-8

    def __init__(self, input_count, output_count, *, activation='ReLU'):
        self.output_count = output_count
        self.input_count = input_count
        self.ppp = output_count
        self.w = np.random.random((input_count, output_count))
        self.b = np.random.random((output_count))
        self.optimize_func = None
        self.v = 0.0
        self.v_sum = 0.0
        self.m = 0.0
        self.m_sum = 0.0
        self.adam_count = 1
        if activation == 'ReLU':
            self.activate = self.activate_relu
        elif activation == 'Sigmoid':
            self.activate = self.activate_sigmoid
        elif activation == 'softmax':
            self.activate = self.activate_softmax

    
    def set_optimizer(self, optimizer=None):
        if optimizer is None:
            self.optimize_func = self.optimize_gradient_descent
        elif optimizer == 'momentum':
            self.optimize_func = self.optimize_momentum
        elif optimizer == 'Adam':
            self.optimize_func = self.optimize_Adam

    
    def update_optimizer_batch(self, bins):
        self.v = self.v_sum / bins
        self.v_sum = 0.0
        self.m = self.m_sum / bins
        self.m_sum = 0.0
        self.adam_count += 1


    def reset(self):
        self.w = np.random.random(self.w.shape)
        self.b = np.random.random(self.b.shape)


    def progress_forword(self, x_input):
        return self.activate(np.matmul(x_input, self.w)+self.b)


    def activate_sigmoid(self, z):
        s = 1 / (1+np.exp(-z))
        return s


    def activate_relu(self, z):
        return np.where(z > 0., z/self.ppp, 0.)

    
    def activate_softmax(self, z):
        z = np.exp(z)
        return z / np.expand_dims(np.sum(z, axis=-1), axis=-1)


    def update_SGD(self, f, fx, learning_rate):
        w_iter = np.nditer(self.w, flags=['multi_index'])

        while not w_iter.finished:
            mi = w_iter.multi_index
            source = self.w[mi]
            dx = 1e-4 * source
            self.w[mi] = source + dx
            f_x_plus_dx = f()
            gradient = (f_x_plus_dx - fx) / dx
            self.w[mi] = source - learning_rate * gradient
            w_iter.iternext()

        source = self.b
        self.b = source + (source*1e-4)
        f_x_plus_dx = f()
        self.b = source - learning_rate * gradient


    def optimize_momentum(self, gradient, learning_rate):
        self.v = Layer.momentum_rate * self.v + learning_rate * gradient
        self.w -= self.v


    def optimize_gradient_descent(self, gradient, learning_rate):
        self.w -= learning_rate * gradient


    def optimize_Adam(self, gradient, learning_rate):
        m = Layer.beta1*self.m + (1-Layer.beta1)*gradient
        v = Layer.beta2*self.v + (1-Layer.beta2)*np.power(gradient,2)
        self.m_sum += m
        self.v_sum += v
        m_hat = m / (1 - np.power(Layer.beta1, self.adam_count))
        v_hat = v / (1 - np.power(Layer.beta2, self.adam_count))
        self.w -= learning_rate * m_hat / (np.sqrt(v_hat) + Layer.epsilon)


class Model:
    def __init__(self):
        self.layers = []
        self.loss = 0.0
        self.stop_count = 0


    def reset(self):
        for layer in self.layers:
            layer.reset()


    def add_layer(self, output_count, input_count=None, *, activation='ReLU'):
        if input_count is None:
            input_count = self.layers[-1].output_count
        self.layers.append(Layer(input_count, output_count, activation=activation))


    def predict(self, x):
        for layer in self.layers:
            x = layer.progress_forword(x)
        return x

    
    def calc_cross_entropy(self, p, y):
        delta = 0.0000001
        return -np.sum(y*np.log(p+delta))

    
    def update_layers(self, x, y, learning_rate):
        fx = self.calc_cross_entropy(self.predict(x), y)
        for layer in self.layers:
            layer.update_SGD(lambda :self.calc_cross_entropy(self.predict(x), y), fx, learning_rate)

    
    def compile(self, loss, *, optimizer=None):
        for layer in self.layers:
            layer.set_optimizer(optimizer)
            layer.reset()
        


    def fit(self, x, y, epochs, learning_rate=0.01, batch_size = None, print_count=None):
        if print_count is None:
            print_count = epochs / 10
        if batch_size is None:
            batch_size = len(y)
        bins, other = divmod(len(y), batch_size)

        for i in range(epochs):
            if i%print_count == 0:
                loss = self.calc_cross_entropy(self.predict(x[:256]), y[:256])
                print(f'ephoc : {i}, loss : {loss}')
            
            s = 0
            for b in range(bins):
                print('>', end='')
                e = s+batch_size
                self.update_layers(x[s:e], y[s:e], learning_rate)
                s = e
            
            for layer in self.layers:
                layer.update_optimizer_batch(bins)
            
            if other != 0:
                self.update_layers(x[s:], y[s:], learning_rate)

            

    def evaluate(self, x, y):
        p = self.predict(x)
        correct_count = 0

        for yi, pi in zip(y, p):
            if yi[0, np.argmax(pi)] == 1.0:
                correct_count += 1

        print(f"accuracy : {correct_count/len(y)}")


data = pd.read_csv("data/mnist_train.csv")
x_train = data.drop(['label'], axis=1).values / 255.0
x_train = x_train.reshape([x_train.shape[0], 1, x_train.shape[-1]])
y_data = data['label']
y_data = pd.get_dummies(y_data)
y_train = y_data.values
y_train = y_train.reshape([y_train.shape[0], 1, y_train.shape[-1]])


test_count = int(len(y_train)/10)
x_test = x_train[-test_count:]
x_train = x_train[:-test_count]
y_test = y_train[-test_count:]
y_train = y_train[:-test_count]

model = Model()
model.add_layer(input_count=784, output_count=256)
model.add_layer(output_count=256)
model.add_layer(output_count=128)
model.add_layer(output_count=64)
model.add_layer(output_count=10, activation='softmax')
model.compile(loss='cross_entropy', optimizer='Adam')
model.fit(x_train, y_train, epochs=10, batch_size=128, learning_rate=0.001)


# data = pd.read_csv("data/mnist_test.csv")
# x_test = data.drop(['label'], axis=1).values / 255.0
# x_test = x_test.reshape([x_test.shape[0], 1, x_test.shape[-1]])
# y_data = data['label']
# y_data = pd.get_dummies(y_data)
# y_test = y_data.values
# y_test = y_test.reshape([y_test.shape[0], 1, y_test.shape[-1]])

model.evaluate(x_test, y_test)
