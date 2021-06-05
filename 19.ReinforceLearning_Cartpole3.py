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

bufferSize = 2048
halfBufferSize = int(bufferSize/2)

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
batchSize = 0
isFirst = True

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

        bufferIndex += 1
        if bufferIndex >= bufferSize:
            bufferIndex = 0

        stepCount += 1

    print("episode: {}  steps: {}".format(i, stepCount))

    if isFirst:
        si = 0
        ei = bufferIndex
        batchSize = min(64,bufferIndex)
        if bufferIndex >halfBufferSize:
            isFirst =False
    elif bufferIndex > halfBufferSize:
        si = 0
        ei = bufferIndex
        batchSize = 64
    else:
        si = halfBufferSize
        ei = bufferSize-1
    
    order = random.sample(orderList[si:ei],batchSize)
    nextOrder = nextOrderList[order]

    Q_target = rewardsBuffer[order] + discountRate * numpy.max(targetModel.predict(
        statesBuffer[nextOrder]), axis=1) * ~terminalsBuffer[order]

    y = mainModel.predict(statesBuffer[order])
    y[numpy.arange(batchSize), actionsBuffer[order]] = Q_target
    mainModel.fit(statesBuffer[order], y, batchSize, verbose=0)

    if temp > 5:
        targetModel.set_weights(mainModel.get_weights())
        temp = 0

    temp += 1

    if stepCount > 150:
        continueCount += 1
        if continueCount > 50:
            print('통과기준을 만족')
            break
    else:
        continueCount = 0

mainModel.save('model/cartpole3.h5')
