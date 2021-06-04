import tensorflow
import numpy
import pandas

trainData = pandas.read_csv('data/titanic_train.csv')

print(trainData.head())
print(trainData.describe())

# 안쓰는 항목 id, 이름과 70% 이상이 n/a인 carbin도 제외한다.
trainData = trainData.drop(['PassengerId', 'Name', 'Cabin', 'Ticket'], axis=1)

# 나이가 없는 사람들의 나이를 각 성별의과 생존 여부로 분류해서 평균으로 한다.
# 생존 여부에 영향을 최대한 줄이기 위해 생존 여부별로 나누어야한다.


def classifyAge(data, survived, sex):
    return data[(data['Survived'] == survived) & (data['Sex'] == sex)]['Age']


ageMean_survivedMale = classifyAge(trainData, 1, 'male').mean()
ageMean_deadMale = classifyAge(trainData, 0, 'male').mean()
ageMean_survivedFemale = classifyAge(trainData, 1, 'female').mean()
ageMean_deadFemale = classifyAge(trainData, 0, 'female').mean()

age1 = classifyAge(trainData, 1, 'male').fillna(ageMean_survivedMale, axis=0)
age2 = classifyAge(trainData, 0, 'male').fillna(ageMean_deadMale, axis=0)
age3 = classifyAge(trainData, 1, 'female').fillna(ageMean_survivedFemale, axis=0)
age4 = classifyAge(trainData, 0, 'female').fillna(ageMean_deadFemale, axis=0)

trainData['Age'] = pandas.concat([age1, age2, age3, age4], axis=0)

print(trainData.isnull().sum())

# 예외가 단 2개만 존재하는 embarked도 삭제한다.
trainData = trainData.dropna(axis=0)

#trainData['Sex'] = trainData['Sex'].apply(lambda x: 1 if x == 'Male' else 0)
# 판다스 제공하는 범주를 자동으로 나눠즈는 함수를 사용하여 문자를 손쉽게 변형
trainData = pandas.get_dummies(trainData, drop_first=True)

# 데이터 노멀라이즈
normalizedTrainData = (trainData-trainData.min()) / (trainData.max()-trainData.min())

# 데이터 섞기
normalizedTrainData = normalizedTrainData.sample(frac=1)

print(normalizedTrainData.head())

x_data = normalizedTrainData.drop('Survived', axis=1).values
y_data = normalizedTrainData['Survived'].values

x_train = numpy.array(x_data[100:])
y_train = numpy.array(y_data[100:])

x_evaluate = numpy.array(x_data[:100])
y_evaluate = numpy.array(y_data[:100])

dropoutRate = 0.3

model = tensorflow.keras.Sequential()

model.add(tensorflow.keras.layers.Dense(
    128, activation='relu', kernel_initializer='glorot_normal'))
model.add(tensorflow.keras.layers.Dropout(dropoutRate))
model.add(tensorflow.keras.layers.Dense(
    256, activation='relu', kernel_initializer='glorot_normal'))
model.add(tensorflow.keras.layers.Dropout(dropoutRate))
model.add(tensorflow.keras.layers.Dense(
    512, activation='relu', kernel_initializer='glorot_normal'))
model.add(tensorflow.keras.layers.Dropout(dropoutRate))
model.add(tensorflow.keras.layers.Dense(
    512, activation='relu', kernel_initializer='glorot_normal'))
model.add(tensorflow.keras.layers.Dropout(dropoutRate))
model.add(tensorflow.keras.layers.Dense(
    256, activation='relu', kernel_initializer='glorot_normal'))
model.add(tensorflow.keras.layers.Dropout(dropoutRate))
model.add(tensorflow.keras.layers.Dense(
    128, activation='relu', kernel_initializer='glorot_normal'))
model.add(tensorflow.keras.layers.Dropout(dropoutRate))
model.add(tensorflow.keras.layers.Dense(
    1, activation='sigmoid', kernel_initializer='glorot_normal'))

model.compile(optimizer=tensorflow.keras.optimizers.Adam(learning_rate=0.0005),
              loss='binary_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=100, epochs=500, validation_split=0.2)


print('학습결과 평가')
evaluations = model.evaluate(x_evaluate, y_evaluate)
print('Loss, Accuracy :', evaluations)


# 테스트 데이터 준비
testData = pandas.read_csv('data/titanic_test.csv')

testData = testData.drop(['PassengerId', 'Name', 'Cabin', 'Ticket'], axis=1)

testData[testData['Fare'].isnull()==True]

ageMean_male = testData[testData['Sex'] == 'male']['Age'].mean()
ageMean_female = testData[testData['Sex'] == 'female']['Age'].mean()
ageMale = testData[testData['Sex'] == 'male']['Age'].fillna(ageMean_male, axis=0)
ageFeMale = testData[testData['Sex'] == 'female']['Age'].fillna(ageMean_female, axis=0)

testData['Age'] = pandas.concat([ageMale, ageFeMale], axis=0)
testData = testData.dropna(axis=0)

testData = pandas.get_dummies(testData, drop_first=True)
normalizedTestData = (testData-testData.min())/(testData.max()-testData.min())

x_test = normalizedTestData.values
# print(normalizedTestData.info())
# normalizedTrainData = normalizedTrainData.drop('Survived', axis=1)
# print(normalizedTrainData.info())

predicts = model.predict(x_test)
num = 0
for i in predicts:
    if i >= 0.5:
        print(num, ': survive')
    else:
        print(num, ': dead')
    num += 1
