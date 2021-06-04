import gym
import random
import tensorflow as tf
import numpy
from tensorflow.python.keras.backend import dtype

environment = gym.make('CartPole-v0')

episodeCount = 1000
discountRate = 0.99
targetInterval = 8  # 몇번마다 한번씩 메인을 타겟에 복사할지
rewardSum = 0

inputSize = environment.observation_space.shape[0]

mainModel = tf.keras.models.Sequential()
mainModel = tf.keras.Sequential()
mainModel.add(tf.keras.layers.Dense(16, 'relu', True, 'glorot_normal'))
mainModel.add(tf.keras.layers.Dense(2, kernel_initializer='glorot_normal'))
mainModel.compile(tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')

targetModel = tf.keras.models.clone_model(mainModel)

bufferSize = 1024

statesBuffer = numpy.zeros([bufferSize, inputSize], dtype=float)
actionsBuffer = numpy.zeros([bufferSize], dtype=int)
rewardsBuffer = numpy.zeros([bufferSize], dtype=int)
terminalsBuffer = numpy.zeros([bufferSize], dtype=bool)
orderList = tuple(range(0, bufferSize))
nextOrderList = numpy.arange(1, bufferSize)

countR = 1 / episodeCount
continueCount = 0  # 몇번 연속 기준을 통과했는지 판단용
temp = 0
bufferIndex = 0

for i in range(episodeCount):
    e = countR * i
    stepCount = 0
    isTerminal = False
    statesBuffer[0] = environment.reset()

    while not isTerminal:
        if random.random() > e:
            actionsBuffer[stepCount] = environment.action_space.sample()
        else:
            x = numpy.reshape(statesBuffer[stepCount], [1, inputSize])
            actionsBuffer[stepCount] = numpy.argmax(mainModel.predict(x))

        statesBuffer[stepCount+1], rewardsBuffer[stepCount], isTerminal, _ = environment.step(
            actionsBuffer[stepCount])

        terminalsBuffer[stepCount] = isTerminal

        if terminalsBuffer[stepCount]:
            rewardsBuffer[stepCount] = -1

        if bufferIndex > 10:
            if bufferIndex > 64:
                batchSize = 64
            else:
                batchSize = bufferIndex

            order = random.sample(orderList[0:bufferIndex], batchSize)
            nextOrder = nextOrderList[order]

            Q_target = rewardsBuffer[0:batchSize] + discountRate * numpy.max(
                targetModel.predict(statesBuffer[1:batchSize+1]), axis=1) * ~terminalsBuffer[0:batchSize]
            y = mainModel.predict(statesBuffer[0:batchSize])
            y[numpy.arange(batchSize), actionsBuffer[0:batchSize]] = Q_target
            mainModel.fit(statesBuffer[0:batchSize], y, batchSize, verbose=0)


        if temp > 10:
            targetModel.set_weights(mainModel.get_weights())
            temp = 0

        bufferIndex += 1
        if bufferIndex >= bufferSize:
            bufferIndex = 0

        temp += 1
        stepCount += 1

    print("episode: {}  steps: {}".format(i, stepCount))

    if stepCount > 150:
        continueCount += 1
        if continueCount > 50:
            print('통과기준을 만족')
            break
    else:
        continueCount = 0

mainModel.save('model/cartpole2.h5')
