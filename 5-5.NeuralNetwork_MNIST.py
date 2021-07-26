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
        self.optimize_func = None
        self.dh_dw = None
        self.da_dh = None
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


    def reset(self, is_output):
        if is_output:
            self.w = np.random.random((self.input_count+1, self.output_count)) / (self.input_count*0.1)
        else:
            self.w = np.random.random((self.input_count+1, self.output_count+1)) / (self.input_count*0.1)
            
            self.w[:, self.output_count] = 0.0
            self.w[self.input_count, self.output_count] = 1.0


    def forword(self, x_input):
        self.dh_dw = x_input.swapaxes(1, 2)
        return self.activate(np.matmul(x_input, self.w))


    def activate_sigmoid(self, z):
        s = 1 / (1+np.exp(-z))
        self.da_dh = s*(1-s)
        return s


    def activate_relu(self, z):
        self.da_dh = np.where(z > 0., 1., 0.)
        return np.where(z > 0., z, 0.)

    
    def activate_softmax(self, z):
        z = np.exp(z)
        return z / np.expand_dims(np.sum(z, axis=-1), axis=-1)


    def backword(self, g, learning_rate):
        g = self.da_dh * g
        return self.calc_backpropagation_and_update(g, learning_rate)


    def calc_backpropagation_and_update(self, gradient, learning_rate):
        temp = self.dh_dw * gradient
        gradient = np.matmul(gradient, self.w.T)
        gradient[:,:,-1] = 0.0
        self.optimize_func(np.sum(temp, axis=0), learning_rate)
        return gradient


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
        x = np.insert(x, x.shape[-1], 1., axis=-1)
        for layer in self.layers:
            x = layer.forword(x)
        return x

    
    def calc_cross_entropy(self, p, y):
        delta = 0.0000001
        return -np.sum(y*np.log(p+delta))

    
    def update_layers(self, x, y, learning_rate, ephoc=None):
        c = self.predict(x)
        if ephoc is not None:
            temp = self.calc_cross_entropy(c, y)
            self.loss += temp
        gradient = c-y#dL / ds
        gradient = self.layers[-1].calc_backpropagation_and_update(gradient, learning_rate)
        other_layers = self.layers[:-1]
        for layer in other_layers[::-1]:
            gradient = layer.backword(gradient, learning_rate)

    
    def compile(self, loss, *, optimizer=None):
        for layer in self.layers[:-1]:
            layer.set_optimizer(optimizer)
            layer.reset(False)

        self.layers[-1].set_optimizer(optimizer)
        self.layers[-1].reset(True)
        


    def fit(self, x, y, epochs, learning_rate=0.01, batch_size = None, print_count=None):
        if print_count is None:
            print_count = epochs / 10
        if batch_size is None:
            batch_size = len(y)
        bins, other = divmod(len(y), batch_size)

        for i in range(epochs):
            s = 0
            self.loss = 0.0
            for b in range(bins):
                e = s+batch_size
                if i % print_count == 0:
                    self.update_layers(x[s:e], y[s:e], learning_rate, i)
                else:
                    self.update_layers(x[s:e], y[s:e], learning_rate)
                s = e
            
            for layer in self.layers:
                layer.update_optimizer_batch(bins)
            
            if other != 0:
                self.update_layers(x[s:], y[s:], learning_rate)
            if i%print_count == 0:
                print(f'ephoc : {i}, loss : {self.loss}')
            

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
model.add_layer(output_count=128)
model.add_layer(output_count=10, activation='softmax')
model.compile(loss='cross_entropy', optimizer='Adam')
model.fit(x_train, y_train, epochs=20, batch_size=128, learning_rate=0.001)


# data = pd.read_csv("data/mnist_test.csv")
# x_test = data.drop(['label'], axis=1).values / 255.0
# x_test = x_test.reshape([x_test.shape[0], 1, x_test.shape[-1]])
# y_data = data['label']
# y_data = pd.get_dummies(y_data)
# y_test = y_data.values
# y_test = y_test.reshape([y_test.shape[0], 1, y_test.shape[-1]])

model.evaluate(x_test, y_test)
