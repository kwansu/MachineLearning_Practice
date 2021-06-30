from NumericalDifferentiation import*
import pandas as pd

#########################데이터 전처리##############################
# kaggle에서 받은자료 출저: https://www.kaggle.com/prathamtripathi/customersegmentation
# 고객의 여러 정보를 토대로 고객의 등급?을 A,B,C,D 4가지로 분류한다.
# 입력 데이터의 종류는 11가지로 나이,성별,급여,지역등 다양하다.
data_pandas = pd.read_csv("data/Telecust1.csv")
data_pandas = data_pandas.dropna(axis=0)

print(data_pandas.head())
print(data_pandas.describe())
# 입력값은 딱히 비거나 특별한 데이터 없이 모두 실수값으로 잘 들어가 있다.
# 결과값 customer category가 A,B,C,D로 나오므로 onehot으로 인코딩해주자.

data_custcat_onehot = pd.get_dummies(data_pandas["custcat"])
print(data_custcat_onehot.head())
print(data_custcat_onehot.describe())

data_pandas = data_pandas.drop("custcat", axis=1)
data_pandas = (data_pandas - data_pandas.min()) /(data_pandas.max() - data_pandas.min())

x_data = data_pandas.values
# x_data = (x_data - np.mean(x_data,axis=0)) / np.std(x_data,axis=0)
y_data = data_custcat_onehot.values
y_data = y_data.astype(float)
###################################################################


def activateSoftmax(z):
    z = np.exp(z)
    temp = z / np.sum(z, axis=-1).reshape(len(z), 1)
    return temp


def hypothesis(x, W, B):
    g = np.dot(x, W) + B
    return activateSoftmax(g)


def crossentropy(P):
    return -np.sum(y_data*np.log(P))

W = np.random.random((x_data.shape[-1], 4))
B = np.random.random(4)
learning_rate = 0.001
x_data_normalized = (x_data - np.mean(x_data)) / np.std(x_data)
cost = lambda _x, _w, _b: crossentropy((hypothesis(_x, _w, _b)))

for i in range(1001):
    if i % 100 == 0:
        print('epoch %d, cost : %f' % (i, cost(x_data_normalized, W, B)))
    W -= (learning_rate * differentiate(lambda t: cost(x_data_normalized, t, B), W))
    B -= (learning_rate * differentiate(lambda t: cost(x_data_normalized, W, t), B))

#print("W : {}, B : {}".format(W, B))


def predict(x):
    #Y = hypothesis(x, W, B)
    Y = np.dot(x, W) + B
    correctCount = 0

    for i in range(len(Y)):
        label = np.argmax(Y[i])
        if label == 0:
            category = 'A'
        elif label == 1:
            category = 'B'
        else:
            category = 'C' if label == 2 else 'D'
        #print("x : {} , predict : {}".format(x[i], category))
        if label == np.argmax(y_data[i]):
            correctCount += 1

    print("accuracy : %f" % (correctCount / len(y_data)))


predict(x_data_normalized)
