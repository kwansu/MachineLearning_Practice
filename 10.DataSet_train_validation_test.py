import numpy
import tensorflow

x_train = numpy.array([[1, 2],
                      [2, 3],
                      [2, 5],
                      [3, 1],
                      [4, 3],
                      [5, 3],
                      [6, 2]], numpy.float)

y_train = numpy.array([[0],
                      [0],
                      [0],
                      [0],
                      [1],
                      [1],
                      [1]], numpy.float)

x_test = numpy.array([[0, 0], [2, 2], [6, 1]], numpy.float)
y_test = numpy.array([[0],[0],[1]], numpy.float)

model = tensorflow.keras.models.Sequential()
model.add(tensorflow.keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=tensorflow.keras.optimizers.SGD(),
              loss='binary_crossentropy', metrics=['accuracy'])

# 학습 데이터의 20%를 검증 데이터로 사용
result = model.fit(x_train, y_train, epochs=3000, validation_split=0.2)

print('학습결과 평가')
evaluation = model.evaluate(x_test, y_test)
print(evaluation)
