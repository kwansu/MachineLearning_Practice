import numpy
import tensorflow

x_data = [[1, 2, 1, 1],
          [2, 1, 3, 2],
          [3, 1, 3, 4],
          [4, 1, 5, 5],
          [1, 7, 5, 5],
          [1, 2, 5, 6],
          [1, 6, 6, 6],
          [1, 7, 7, 7]]

y_data = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]]

model = tensorflow.keras.models.Sequential()
model.add(tensorflow.keras.layers.Dense(3, activation='softmax')) # softmax에 대해 더 알아보기

# categorical_crossentropy => -∑(L * log(y))
model.compile(optimizer=tensorflow.keras.optimizers.SGD(learning_rate=0.1),
              loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(numpy.array(x_data, numpy.float), numpy.array(y_data, numpy.float), epochs=1500)

predict = model.predict(numpy.array([[1, 11, 7, 9]]))
print(predict)
