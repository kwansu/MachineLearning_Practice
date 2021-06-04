import numpy
import tensorflow

loadedData = numpy.loadtxt('data/data-01-test-score.csv', delimiter=',', dtype=numpy.float32)
x_data = loadedData[:, 0:-1]
y_data = loadedData[:, [-1]]

model = tensorflow.keras.models.Sequential([tensorflow.keras.layers.Dense(1, activation='linear')])

model.compile(optimizer = tensorflow.keras.optimizers.SGD(lr=1e-5), loss='mse')

model.fit(x_data, y_data, epochs=1000)

predict = model.predict(numpy.array([[80, 80, 80]]))
print(predict)
