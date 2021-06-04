import tensorflow

mnist = tensorflow.keras.datasets.mnist

# 28x28 사이즈의 손글씨 데이타베이스 mnist를 로드한다. train 60000, test 10000
(x_train, y_train), (x_test, y_test) = tensorflow.keras.datasets.mnist.load_data()

# 각 픽셀당 1바이트 0~255의 크기 이므로 일단 정규화를 시킨다.
# 근대, 어차피 모든 픽셀의 정보의 범위가 같아서 필요없을지도?
# 결국 정규화를 하는 것이 더 좋은 결과가 나왔다. bios값과 w의 차이 때문인듯?
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])

y_train = tensorflow.keras.utils.to_categorical(y_train, 10)
y_test = tensorflow.keras.utils.to_categorical(y_test, 10)

model = tensorflow.keras.models.Sequential()
model.add(tensorflow.keras.layers.Dense(10, activation='softmax'))
model.compile(optimizer=tensorflow.keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=100, epochs=20, validation_split=0.1)

print('학습결과 평가')
predictions = model.evaluate(x_test, y_test)
print('Loss, Accuracy :', predictions)
