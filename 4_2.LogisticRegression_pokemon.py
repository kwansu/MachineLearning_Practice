from NumericalDifferentiation import differentiate
from collections import Counter
import tensorflow as tf
import pandas as pd
import numpy as np
import re


#########################데이터 전처리##############################
# kaggle에서 받은자료 출저: https://www.kaggle.com/abcsds/pokemon
pokemon = pd.read_csv("data/Pokemon.csv")

# 이름 문자수와 문자가 10개 넘는 여부
pokemon["name_count"] = pokemon["Name"].apply(lambda i: len(i))
pokemon["long_name"] = pokemon["name_count"].apply(lambda i: 1.0 if i >= 10 else 0.0)

# 이름에 공백을 제거하고 영문자 아닌 예외 처리
pokemon["Name_nospace"] = pokemon["Name"].apply(lambda i: i.replace(" ", ""))
pokemon["name_isalpha"] = pokemon["Name_nospace"].apply(lambda i: i.isalpha())
pokemon[pokemon["name_isalpha"] == False]
pokemon = pokemon.replace(to_replace="Nidoran♀", value="Nidoran X")
pokemon = pokemon.replace(to_replace="Nidoran♂", value="Nidoran Y")
pokemon = pokemon.replace(to_replace="Farfetch'd", value="Farfetchd")
pokemon = pokemon.replace(to_replace="Mr. Mime", value="Mr Mime")
pokemon = pokemon.replace(to_replace="Porygon2", value="PorygonTwo")
pokemon = pokemon.replace(to_replace="Ho-oh", value="Ho Oh")
pokemon = pokemon.replace(to_replace="Mime Jr.", value="Mime Jr")
pokemon = pokemon.replace(to_replace="Porygon-Z", value="Porygon Z")
pokemon = pokemon.replace(to_replace="Zygarde50% Forme", value="Zygarde Forme")
pokemon.loc[[34, 37, 90, 131, 252, 270, 487, 525, 794]]
pokemon["Name_nospace"] = pokemon["Name"].apply(lambda i: i.replace(" ", ""))
pokemon["name_isalpha"] = pokemon["Name_nospace"].apply(lambda i: i.isalpha())
pokemon[pokemon["name_isalpha"] == False]

def tokenize(name): # 이름을 토큰화 시키는 함수
    name_split = name.split(" ")
    
    tokens = []
    for part_name in name_split:
        a = re.findall('[A-Z][a-z]*', part_name)
        tokens.extend(a)
        
    return np.array(tokens)

# 전설 포켓몬의 이름을 토큰화
legendary = pokemon[pokemon["Legendary"] == True].reset_index(drop=True)
all_tokens = list(legendary["Name"].apply(tokenize).values)
token_set = []
for token in all_tokens:
    token_set.extend(token)

# 토큰화 문자들 중 많이 사용하는 토큰 10개를 뽑는다.
most_common = Counter(token_set).most_common(10)

# 10개의 토큰에 포함되는 문자 가있는지 여부를 추가
for token, _ in most_common:
    temp = pokemon["Name"].str.contains(token)
    pokemon[token] = temp.apply(lambda i: 1.0 if i else 0.0)

types = list(set(pokemon["Type 1"])) # 모든 타입 종류

for t in types: # 범주형인 타입을 one_hot 인코딩으로 변환
    pokemon[t] = (pokemon["Type 1"] == t) | (pokemon["Type 2"] == t)

# 사용하는 특성들
features = ['Total', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Generation', 
            'name_count', 'long_name', 'Forme', 'Mega', 'Mewtwo', 'Kyurem', 'Deoxys', 'Hoopa', 
            'Latias', 'Latios', 'Kyogre', 'Groudon', 'Poison', 'Water', 'Steel', 'Grass', 
            'Bug', 'Normal', 'Fire', 'Fighting', 'Electric', 'Psychic', 'Ghost', 'Ice', 
            'Rock', 'Dark', 'Flying', 'Ground', 'Dragon', 'Fairy']

# 수치가 큰 값들을 정규화 시켜준다.
big_features = ['Total', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
n_columns = pokemon[big_features]
n_columns = n_columns / n_columns.max() # min = 0으로 하고 정규화
temp = pokemon[features]
temp[big_features] = n_columns
print(temp.describe())
x_data = temp.to_numpy(dtype=float)


y_data = pokemon['Legendary'].to_numpy(dtype=float)
###################################################################


def activate_sigmoid(z):
    return 1 / (1+np.exp(-z))


def hypothesis(X, W, b):
    g = np.dot(X, W)/len(W) + b
    return activate_sigmoid(g)


def binaryCrossentropy(p, y):
    delta = 0.0000001
    return -np.sum(y*np.log(p+delta) + (1-y)*np.log(1-p+delta))/800


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
    if i % 10 == 0:
        print(f'ephoc : {i}, loss : {calculate_loss(x_data, y_data, W, b)}')
    temp = differentiate(lambda t: calculate_loss(x_data, y_data, t, b), W)
    W -= learningRate * temp
    b -= learningRate * differentiate(lambda t: calculate_loss(x_data, y_data, W, t), b)

evaluate(x_data)
