import numpy as np


class Layer:
    def __init__(self, input_shape, output_shape):
        self.output_shape = output_shape
        self.w = np.random.random(input_shape, output_shape)
        self.b = np.random.random(output_shape)


    def progress(self, x_input):
        return self.activate_sigmoid(x_input * self.w + self.b)


    def activate_sigmoid(z):
        return 1 / (1+np.exp(-z))

    
    def __iter__(self):
        return self

    
    def __next__(self):
        pass



class Model:
    def __init__(self):
        self.layers = []


    def add_layer(self, output_shape, input_shape=None):
        if input_shape is None:
            input_shape = self.layers[-1].output_shape
        self.layers.append(Layer(input_shape, output_shape))

    
    def predict(self, x):
        p = x
        for layer in self.layers:
            p = layer.progress(p)
        return p

    
    def calc_loss(self, x, y):
        p = self.predict(x)
        delta = 0.0000001
        return -np.sum(y*np.log(p+delta) + (1-y)*np.log(1-p+delta))


    def calc_gradient(self, x, y):
        pass

    
    def fit(self, x, y, epochs, learning_rate):
        print_count = epochs / 10
        for i in range(epochs):
            if i % print_count == 0:
                print(f'ephoc : {i}, loss : {self.calc_loss(x, y)}')
            w -= (learning_rate * differentiate(lambda t: self.calc_loss(x, y), w))
            b -= (learning_rate * differentiate(lambda t: self.calc_loss(x, y), b))
        
        
            


layer1 = Layer([2, 4])
layer2 = Layer([4, 1])


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
