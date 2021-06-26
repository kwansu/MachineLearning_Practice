from numpy.core.fromnumeric import reshape
from NumericalDifferentiation import*

loadedData = np.loadtxt('data/linear_multiFeature.csv', delimiter=',', dtype=np.float32)
x_data = loadedData[:, 0:-1]
y_data = loadedData[:, [-1]]

def hypothesisFunction(x, W, b):
    return np.dot(x, W) + b

def costFunction(x, W,b):
    predict = hypothesisFunction(x,W,b)
    temp = (y_data - predict)**2
    return np.sum(temp)

W = np.random.random(3)
W = np.reshape(W,[3,1])
b = np.random.random(1)

def predict(x):
    y = hypothesisFunction(x,W,b)
    print("predict {} : {}".format(x, y))

for i in range(1000):
    print('cost : %f' % costFunction(x_data,W,b))
    t1 =  numerical_derivative(lambda t:costFunction(x_data,t,b),W)
    W -= 0.0000001 * t1
    b -= 0.0000001 *numerical_derivative(lambda t:costFunction(x_data,W,t),b)

print("W : {}, b : {}".format(W, b))
predict((90,90,90))
