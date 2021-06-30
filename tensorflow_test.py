import tensorflow as tf
import pandas as pd
import numpy as np


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


model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Dense(16, activation='relu'))
# model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(4, activation='softmax'))
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(np.array(x_data, np.float), np.array(y_data, np.float), epochs=100)
