from NumericalDifferentiation import*

x_data = np.array((1,2,3,4,5))
y_data = np.array((3,1,-1,-3,-5))


def hypothesis(x, w, b):
    return x*w + b


def activateMeanSquaredError(y):
    return sum((y_data - y)**2)


w = np.random.random(1)
b = np.random.random(1)
cost = lambda _x,_w,_b: activateMeanSquaredError(hypothesis(_x,_w,_b))

for i in range(1001):
    if i%100 == 0:
        print('ephoc %d, cost : %f' %(i, cost(x_data,w,b)))
    w -= 0.01 * numerical_derivative(lambda t:cost(x_data,t,b),w)
    b -= 0.01 * numerical_derivative(lambda t:cost(x_data,w,t),b)

print("w : {}, b : {}".format(w, b))
print("x : 4, predict : %f" %hypothesis(4,w,b))
