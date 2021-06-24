from NumericalDifferentiation import*

x_data = np.array((1,2,3,4,5))
y_data = np.array((3,1,-1,-3,-5))

w = np.random.random(1)
b = np.random.random(1)

def hypothesisFunction(x, w, b):
    return x*w + b

def costFunction(x, w,b):
    return sum((y_data - hypothesisFunction(x,w,b))**2)

for i in range(1000):
    print(costFunction(x_data,w,b))
    w -= 0.01 * y_data(lambda t:costFunction(x_data,t,b),w)
    b -= 0.01 *numerical_derivative(lambda t:costFunction(x_data,y_data,t),b)
y_data
print("w : {}, b : {}".format(w, b))
print(hypothesisFunction(4,w,b))
