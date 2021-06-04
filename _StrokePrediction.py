import tensorflow
import numpy
import pandas

# 다시보니 데이터들의 nomalize, validation 등 빠진게 많다. 다시 한번 처리해보자.

# data preprocessing
loadedData = pandas.read_csv('data/healthcare-dataset-stroke-data.csv')

loadedData.drop(columns=['id'], inplace=True)

# print(loadedData.isnull().sum())
loadedData = loadedData.dropna()
loadedData.drop(columns=['smoking_status', 'work_type'], inplace=True)

# 문자열 데이터들 분류
loadedData['gender'] = loadedData['gender'].apply(
    lambda x: 1 if x == 'Male' else 0)
loadedData["Residence_type"] = loadedData["Residence_type"].apply(
    lambda x: 1 if x == "Urban" else 0)
loadedData["ever_married"] = loadedData["ever_married"].apply(
    lambda x: 1 if x == "Yes" else 0)

loadedData.info()

# 데이터 섞기
loadedData = loadedData.sample(frac=1)

x_data = []
y_data = loadedData['stroke'].values
loadedData.drop(columns=['stroke'], inplace=True)

for i, rows in loadedData.iterrows():
    x_data.append([rows['gender'], rows['age'], rows['hypertension'], rows['ever_married'],
                  rows['heart_disease'], rows['Residence_type'], rows['avg_glucose_level'], rows['bmi']])

x_train = numpy.array(x_data[200:])
y_train = numpy.array(y_data[200:])

x_test = numpy.array(x_data[:200])
y_test = numpy.array(y_data[:200])

model = tensorflow.keras.models.Sequential()

model.add(tensorflow.keras.layers.Dense(81, activation='relu'))
model.add(tensorflow.keras.layers.Dense(729, activation='relu'))
model.add(tensorflow.keras.layers.Dense(729, activation='relu'))
model.add(tensorflow.keras.layers.Dense(81, activation='relu'))
model.add(tensorflow.keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=tensorflow.keras.optimizers.Adam(learning_rate=0.01),
              loss='binary_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=100, validation_split=0.2)

print('학습결과 평가')
evaluation = model.evaluate(x_test, y_test)
print('loss: ', evaluation[0])
print('accuracy', evaluation[1])
