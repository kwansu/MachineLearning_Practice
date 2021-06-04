import numpy
import tensorflow

x_data = numpy.array([[73., 80., 75.],
                      [93., 88., 93.],
                      [89., 91., 90.],
                      [96., 98., 100.],
                      [73., 66., 70.]], numpy.float)

y_data = numpy.array([152., 185., 180., 196., 142.], numpy.float)

model = tensorflow.keras.models.Sequential(
    [tensorflow.keras.layers.Dense(1, activation='linear')])

model.compile(optimizer=tensorflow.keras.optimizers.SGD(learning_rate=0.00001), loss='mse')

model.fit(x_data, y_data, epochs=1000)

predict = model.predict(numpy.array([[100, 100, 80]]))
print(predict)
