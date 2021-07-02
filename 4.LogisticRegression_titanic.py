from NumericalDifferentiation import differentiate
import numpy as np
import pandas as pd


################################데이터 전처리#################################
def preprocessData(data, isTrain=True):
    # 안쓰는 항목 id, 이름과 70% 이상이 n/a인 carbin도 제외한다.
    data = data.drop(["PassengerId", "Name", "Cabin", "Ticket"], axis=1)

    # print(data.describe())

    if isTrain:
        # 나이가 없는 사람들의 나이를 각 성별의과 생존 여부로 분류해서 평균으로 한다.
        # 생존 여부에 영향을 최대한 줄이기 위해 생존 여부별로 나누어야한다.
        def classifyAge(data, survived, sex): return data[(
            data["Survived"] == survived) & (data["Sex"] == sex)]["Age"]

        age_mean_survived_male = classifyAge(data, 1, "male").mean()
        age_mean_dead_male = classifyAge(data, 0, "male").mean()
        age_mean_survived_female = classifyAge(data, 1, "female").mean()
        age_mean_dead_female = classifyAge(data, 0, "female").mean()

        age1 = classifyAge(data, 1, "male").fillna(
            age_mean_survived_male, axis=0)
        age2 = classifyAge(data, 0, "male").fillna(age_mean_dead_male, axis=0)
        age3 = classifyAge(data, 1, "female").fillna(
            age_mean_survived_female, axis=0)
        age4 = classifyAge(data, 0, "female").fillna(
            age_mean_dead_female, axis=0)
        data["Age"] = pd.concat([age1, age2, age3, age4], axis=0)
    else:
        mean = data["Age"].mean()
        data["Age"] = data["Age"].fillna(mean, axis=0)

    # 예외가 단 2개만 존재하는 embarked도 삭제한다.
    data = data.dropna(axis=0)

    # data['Sex'] = data['Sex'].apply(lambda x: 1 if x == 'Male' else 0)
    # 판다스 제공하는 범주를 자동으로 나눠즈는 함수를 사용하여 문자를 손쉽게 변형
    data = pd.get_dummies(data, drop_first=True)

    # # 데이터 표준화
    # train_data_normalized = (data - data.mean()) / data.std()
    # print(train_data_normalized.head())

    return data


# kaggle의 타이타닉 생존자 문제에 대한 데이터이다.
data_train = pd.read_csv("data/titanic_train.csv")
data_test = pd.read_csv("data/titanic_test.csv")

data_train_preprocessed = preprocessData(data_train)
x_data = data_train_preprocessed.drop("Survived", axis=1).values
x_data = (x_data - np.mean(x_data, axis=0)) / np.std(x_data, axis=0)
y_data = data_train_preprocessed["Survived"].values
y_data = np.reshape(y_data, [len(y_data), 1])
##########################################################################


def activate_sigmoid(z):
    return 1 / (1+np.exp(-z))


def hypothesis(X, W, b):
    g = np.dot(X, W) + b
    return activate_sigmoid(g)


def binaryCrossentropy(p, y):
    delta = 0.0000001
    return -np.sum(y*np.log(p+delta) + (1-y)*np.log(1-p+delta))


def calculate_loss(x, y, w, b):
    return binaryCrossentropy(hypothesis(x, w, b), y)


def predict(x):
    H = hypothesis(x, W, b)
    return np.array([([1.0] if h >= 0.5 else [0.0]) for h in H])


def evaluate(x):
    e = np.logical_not(np.logical_xor(y_data, predict(x)))
    print(f"accuracy : {np.sum(e)/e.size}")


W = np.random.random((len(x_data[0]), 1))
b = np.random.random(1)
learningRate = 0.001

for i in range(1001):
    if i % 100 == 0:
        print(f'ephoc : {i}, loss : {calculate_loss(x_data, y_data, W, b)}')
    W -= learningRate * differentiate(lambda t: calculate_loss(x_data, y_data, t, b), W)
    b -= learningRate * differentiate(lambda t: calculate_loss(x_data, y_data, W, t), b)

evaluate(x_data)


# test
data_test_preprocessed = preprocessData(data_test, False)
x_test = data_test_preprocessed.values
x_test = (x_test - np.mean(x_test, axis=0)) / np.std(x_test, axis=0)

y_predict = predict(x_test)

for i, p in enumerate(y_predict):
    #print(f"{i} : {'survive' if p else 'dead'}")
    pass
