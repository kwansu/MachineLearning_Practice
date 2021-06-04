import tensorflow
import numpy
import pandas
from sklearn import preprocessing

# data preprocessing
loadedData = pandas.read_csv('data/data.csv')

# 값의 변화가 없는 항목은 날린다. min == max가 같아서 정규화시에 나누기 0으로 문제발생
for columnName in loadedData.columns:
    if (loadedData[columnName].min() == loadedData[columnName].max()):
        loadedData.drop(columns=[columnName], inplace=True)

normalizedData=(loadedData-loadedData.min())/(loadedData.max()-loadedData.min())

print(normalizedData.info())
print(normalizedData.describe())
# 데이터 섞기
normalizedData = normalizedData.sample(frac=1)

y_data = normalizedData['Bankrupt?'].values
normalizedData.drop(columns=['Bankrupt?'], inplace=True)
x_data = normalizedData.values

x_train = numpy.array(x_data[500:])
y_train = numpy.array(y_data[500:])

x_test = numpy.array(x_data[:500])
y_test = numpy.array(y_data[:500])

dropoutRate = 0.3

model = tensorflow.keras.Sequential()
model.add(tensorflow.keras.layers.Dense(190, activation='relu', kernel_initializer='glorot_normal'))
model.add(tensorflow.keras.layers.Dropout(dropoutRate))
model.add(tensorflow.keras.layers.Dense(380, activation='relu', kernel_initializer='glorot_normal'))
model.add(tensorflow.keras.layers.Dropout(dropoutRate))
model.add(tensorflow.keras.layers.Dense(380, activation='relu', kernel_initializer='glorot_normal'))
model.add(tensorflow.keras.layers.Dropout(dropoutRate))
model.add(tensorflow.keras.layers.Dense(190, activation='relu', kernel_initializer='glorot_normal'))
model.add(tensorflow.keras.layers.Dropout(dropoutRate))
model.add(tensorflow.keras.layers.Dense(95, activation='relu', kernel_initializer='glorot_normal'))
model.add(tensorflow.keras.layers.Dropout(dropoutRate))
model.add(tensorflow.keras.layers.Dense(1, activation='sigmoid', kernel_initializer='glorot_normal'))
model.compile(optimizer=tensorflow.keras.optimizers.Adam(learning_rate=0.0001),
              loss='binary_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=100, epochs=500, validation_split=0.2)

print('학습결과 평가')
predictions = model.evaluate(x_test, y_test)
print('Loss, Accuracy :', predictions)