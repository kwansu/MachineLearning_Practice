import numpy
import tensorflow

x_data = numpy.array([[1, 2],
                      [2, 3],
                      [3, 1],
                      [4, 3],
                      [5, 3],
                      [6, 2]], numpy.float)

y_data = numpy.array([[0],
                      [0],
                      [0],
                      [1],
                      [1],
                      [1]], numpy.float)

model = tensorflow.keras.models.Sequential()
model.add(tensorflow.keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=tensorflow.keras.optimizers.SGD(),
              loss='binary_crossentropy', metrics=['accuracy'])

model.fit(x_data, y_data, epochs=1000)  # enpochs = 5000
# 반복 횟수를 4000을 넘도록 크게 하면 정답률이 1.0이 되는데, 이건 오버피팅인듯하다.
