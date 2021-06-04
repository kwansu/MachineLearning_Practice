import numpy
import tensorflow

x_train = [1, 2, 3, 4]
y_train = [2, 0, -2, -4]

model = tensorflow.keras.models.Sequential([tensorflow.keras.layers.Dense(1, activation='linear')])

model.compile(optimizer=tensorflow.keras.optimizers.SGD(learning_rate=0.1), loss='mse')

model.fit(numpy.array(x_train, numpy.float), numpy.array(y_train, numpy.float), epochs=1000)

predict = model.predict([-2, 7])
print(predict)

# w = tensorflow.Variable(tensorflow.random.normal([1]))
# b = tensorflow.Variable(tensorflow.random.normal([1]))

# def loss_function(x, y):
#     return tensorflow.square(y - (w * x - b))

# opt = tensorflow.optimizers.Adam()

# for dummy in range(500):
#     for i in range(0, len(x_train)):
#         opt.minimize(loss_function, var_list = [w, b])
#         print(w.numpy(), b.numpy())


