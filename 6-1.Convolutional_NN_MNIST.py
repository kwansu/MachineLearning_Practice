import pandas as pd
import numpy as np


class Layer:
    beta1 = 0.9
    beta2 = 0.999
    momentum_rate = 0.9
    epsilon = 1e-8

    def __init__(self, input_shape, output_shape, *, activation='ReLU'):
        self.output_shape = output_shape
        self.input_shape = input_shape
        self.optimize_func = None
        self.dh_dw = None
        self.do_dh = None
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


    def setup(self):
        pass

    def progress(self, x_input):
        self.dh_dw = x_input.swapaxes(1, 2)
        return self.activate(np.matmul(x_input, self.w))


    def activate_sigmoid(self, z):
        s = 1 / (1+np.exp(-z))
        self.do_dh = s*(1-s)
        return s


    def activate_relu(self, z):
        self.do_dh = np.where(z > 0., 1., 0.)
        return np.where(z > 0., z, 0.)

    
    def activate_softmax(self, z):
        z = np.exp(z)
        return z / np.expand_dims(np.sum(z, axis=-1), axis=-1)


    def update_backpropagation(self, g, learning_rate):
        g = self.do_dh * g
        return self.calc_backpropagation_and_update(g, learning_rate)


    def calc_backpropagation_and_update(self, gradient, learning_rate):
        temp = np.matmul(self.dh_dw, gradient)
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


class ConvolutionLayer(Layer):
    def __init__(self, filters, kernel, input_shape, *, strid=1, activation):
        super().__init__(input_shape, output_shape=None, activation=activation)
        self.filters = filters
        self.kernel = kernel

        # output 계산
        temp = ((np.array(input_shape) - np.array(kernel+(1,)))
                 / np.array(strid) + 1).astype(np.int64)
        temp[-1] = filters
        
        self.output_shape = tuple(temp)


    def setup(self):
        self.w = np.random.random((self.filters,) + self.kernel)
        self.b = np.random.random((self.filters, self.kernel[-1]))


    def calc_convolutional_filter(self, x_input, filter):
        filter = np.flipud(np.fliplr(filter))
        sub_matrices = np.lib.stride_tricks.as_strided(x_input,
                        shape = tuple(np.subtract(x_input.shape[1:], filter.shape))+filter.shape, 
                        strides = self.strides * 2)

        return np.einsum('ij,klij->kl', filter, sub_matrices)


    def progress(self, x_input):
        temp = self.calc_convolutional_filter(x_input, self.w[0])
        a = None
        return temp


class Model:
    def __init__(self):
        self.layers = []
        self.loss = 0.0
        self.stop_count = 0


    def reset(self):
        for layer in self.layers:
            layer.reset()


    def add_layer(self, output_shape, input_shape=None, *, activation='ReLU'):
        if input_shape is None:
            input_shape = self.layers[-1].output_shape
        self.layers.append(Layer(input_shape, output_shape, activation=activation))


    def add_convolution_layer(self, filters, kernel, input_shape=None, *, strid=1, activation='ReLU'):
        if input_shape is None:
            input_shape = self.layers[-1].output_shape
        self.layers.append(ConvolutionLayer(filters, kernel, input_shape, strid=strid, activation=activation))


    def predict(self, x):
        for layer in self.layers:
            x = layer.progress(x)
        return x

    
    def calc_cross_entropy(self, p, y):
        delta = 0.0000001
        return -np.sum(y*np.log(p+delta))

    
    def update_layers(self, x, y, learning_rate, ephoc=None):
        c = self.predict(x)
        if ephoc is not None:
            temp = self.calc_cross_entropy(c, y)
            self.loss += temp
        gradient = c-y
        gradient = self.layers[-1].calc_backpropagation_and_update(gradient, learning_rate)
        other_layers = self.layers[:-1]
        for layer in other_layers[::-1]:
            gradient = layer.update_backpropagation(gradient, learning_rate)

    
    def compile(self, loss, *, optimizer=None):
        for layer in self.layers:
            layer.set_optimizer(optimizer)
            layer.setup()
        

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
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
y_data = data['label']
y_data = pd.get_dummies(y_data)
y_train = y_data.values

test_count = int(len(y_train)/10)
x_test = x_train[-test_count:]
x_train = x_train[:-test_count]
y_test = y_train[-test_count:]
y_train = y_train[:-test_count]

model = Model()
model.add_convolution_layer(16, (3,3), input_shape=(28,28,1))
model.add_convolution_layer(32, (3,3))
model.add_layer(output_shape=128)
model.add_layer(output_shape=10, activation='softmax')
model.compile(loss='cross_entropy', optimizer='Adam')

model.fit(x_train, y_train, epochs=20, batch_size=128, learning_rate=0.001)


# data = pd.read_csv("data/mnist_test.csv")
# x_test = data.drop(['label'], axis=1).values / 255.0
# y_data = data['label']
# y_data = pd.get_dummies(y_data)
# y_test = y_data.values

model.evaluate(x_test, y_test)
