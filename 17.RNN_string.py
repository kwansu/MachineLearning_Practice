import tensorflow
import numpy

# hihello
x_data = numpy.array([[[1, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0],
                       [1, 0, 0, 0, 0],
                       [0, 0, 1, 0, 0],
                       [0, 0, 0, 1, 0],
                       [0, 0, 0, 1, 0]]], numpy.float)

y_data = [[1, 0, 2, 3, 3, 4]]
y_data = tensorflow.keras.utils.to_categorical(y_data, num_classes=5)

model = tensorflow.keras.Sequential()
cell = tensorflow.keras.layers.LSTMCell(5)
model.add(tensorflow.keras.layers.RNN(cell=cell, return_sequences=True))
model.add(tensorflow.keras.layers.TimeDistributed(
    tensorflow.keras.layers.Dense(5, activation='softmax')))
model.compile(optimizer=tensorflow.keras.optimizers.Adam(
    learning_rate=0.1), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_data, y_data, epochs=50)

vocabularies = ['h', 'i', 'e', 'l', 'o']

predictions = model.predict(x_data)
for i, prediction in enumerate(predictions):
    print([vocabularies[c] for c in numpy.argmax(prediction, axis=1)])
