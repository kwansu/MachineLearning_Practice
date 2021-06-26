from NumericalDifferentiation import*
import pandas

###########################데이터 전처리##############################
# kaggle의 타이타닉 생존자 문제에 대한 데이터이다.
train_data = pandas.read_csv("data/titanic_train.csv")

# 안쓰는 항목 id, 이름과 70% 이상이 n/a인 carbin도 제외한다.
train_data = train_data.drop(["PassengerId", "Name", "Cabin", "Ticket"], axis=1)

# 나이가 없는 사람들의 나이를 각 성별의과 생존 여부로 분류해서 평균으로 한다.
# 생존 여부에 영향을 최대한 줄이기 위해 생존 여부별로 나누어야한다.
classifyAge = lambda data, survived, sex: data[(data["Survived"] == survived) & (data["Sex"] == sex)]["Age"]

ageMean_survivedMale = classifyAge(train_data, 1, "male").mean()
ageMean_deadMale = classifyAge(train_data, 0, "male").mean()
ageMean_survivedFemale = classifyAge(train_data, 1, "female").mean()
ageMean_deadFemale = classifyAge(train_data, 0, "female").mean()

age1 = classifyAge(train_data, 1, "male").fillna(ageMean_survivedMale, axis=0)
age2 = classifyAge(train_data, 0, "male").fillna(ageMean_deadMale, axis=0)
age3 = classifyAge(train_data, 1, "female").fillna(
    ageMean_survivedFemale, axis=0)
age4 = classifyAge(train_data, 0, "female").fillna(ageMean_deadFemale, axis=0)

train_data["Age"] = pandas.concat([age1, age2, age3, age4], axis=0)

# 예외가 단 2개만 존재하는 embarked도 삭제한다.
train_data = train_data.dropna(axis=0)

# train_data['Sex'] = train_data['Sex'].apply(lambda x: 1 if x == 'Male' else 0)
# 판다스 제공하는 범주를 자동으로 나눠즈는 함수를 사용하여 문자를 손쉽게 변형
train_data = pandas.get_dummies(train_data, drop_first=True)

# 데이터 표준화
train_data_normalized = (train_data - train_data.mean()) / train_data.std()
print(train_data_normalized.head())

x_data = train_data_normalized.drop("Survived", axis=1).values
y_data = train_data_normalized["Survived"].values
##########################################################################


def hypothesisFunction(x, w, b):
    g = np.dot(x, w) + b
    temp = sigmoidFunction(g)
    return temp


def costFunction(x, w, b):
    h = hypothesisFunction(x, w, b)
    delta = 1e-7
    temp = y_data*np.log(h+delta) + (1-y_data)*np.log(1-h+delta)
    return -np.sum(temp)


W = np.array([1.])
W = np.reshape(W, [1, 1])
b = np.array([1.])


def predict(x):
    y = hypothesisFunction(x, W, b)
    iterator = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    _x = np.reshape(x_data, [11, 1])
    sameCount = 0

    while not iterator.finished:
        iterIndex = iterator.multi_index
        predic_value = True if y[iterIndex] >= 0.5 else False
        print("x : {} , predict : {}".format(_x[iterIndex], predic_value))
        if predic_value == bool(y_data[iterIndex]):
            sameCount += 1
        iterator.iternext()

    print("accuracy : %f" % (sameCount / y_data.size))


learningRate = 0.01

for i in range(1000):
    print('epoch %d, cost : %f' %
          (i, costFunction(x_data_normalization, W, b)))
    t1 = numerical_derivative(
        lambda t: costFunction(x_data_normalization, t, b), W)
    W -= learningRate * t1
    b -= learningRate * \
        numerical_derivative(lambda t: costFunction(
            x_data_normalization, W, t), b)

print("W : {}, b : {}".format(W, b))
predict(x_data_normalization)
