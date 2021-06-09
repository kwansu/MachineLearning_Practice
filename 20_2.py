from python_simulation.MainPyGame import MainPygame
from AvoidPoop_World import World_AvoidPoop as World
import tensorflow as tf
import numpy as np
import collections
import random
import threading


episodeCount = 500
discountRate = 0.99

mainModel = tf.keras.models.Sequential()
mainModel = tf.keras.Sequential()
mainModel.add(tf.keras.layers.Dense(961, tf.nn.relu, True, 'glorot_normal'))
mainModel.add(tf.keras.layers.Dense(961, tf.nn.relu, True, 'glorot_normal'))
mainModel.add(tf.keras.layers.Dense(961, tf.nn.relu, True, 'glorot_normal'))
mainModel.add(tf.keras.layers.Dense(31, tf.nn.relu, True, 'glorot_normal'))
mainModel.add(tf.keras.layers.Dense(3, kernel_initializer='glorot_normal'))
mainModel.compile(tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')

targetModel = tf.keras.models.clone_model(mainModel)

countR = 1 / episodeCount
continueCount = 0  # 몇번 연속 기준을 통과했는지 판단용
targetCount = targetInterval-batchSize

width = 300
height = 400
world = World(width, height, targetModel)


def runSimulation():
    simulation = MainPygame(width=width, height=height, speed=1, fps=5)
    simulation.run(world)

simulationThread = threading.Thread(target=runSimulation)
simulationThread.start()

batchSize = 64
targetInterval = 10
bufferSize = 10000
buffer = collections.deque(maxlen=bufferSize)
statesBuffer = np.zeros([bufferSize+1, 900])
bufferIndex = 0

for i in range(episodeCount):
    e = 1. / ((i / 10) + 1)
    #e = countR * i
    isTerminal = False
    stepCount = 0
    rewardSum = 0
    world.setupStepSimulation(statesBuffer[bufferIndex])

    while not isTerminal:
        if random.random() < e:
            action = random.randrange(0, 3)
        else:
            x = np.reshape(statesBuffer[bufferIndex], [1, 900])
            action = np.argmax(mainModel.predict(x))

        reward, isTerminal = world.step(action, statesBuffer[bufferIndex+1])

        buffer.append([statesBuffer[bufferIndex], action, reward, statesBuffer[bufferIndex+1], isTerminal])

        if len(buffer) > batchSize:
            trainBatch = random.sample(buffer, batchSize)
            states = np.array([x[0] for x in trainBatch])
            actions = np.array([x[1] for x in trainBatch])
            rewards = np.array([x[2] for x in trainBatch])
            nextStates = np.array([x[3] for x in trainBatch])
            terminals = np.array([x[4] for x in trainBatch])

            Q_target = rewards + discountRate * \
                np.max(targetModel.predict(nextStates), axis=1) * ~terminals

            y = mainModel.predict(states)
            y[np.arange(len(states)), actions] = Q_target

            mainModel.fit(states, y, batchSize, verbose=0)

        if targetCount > targetInterval:
            targetCount = 0
            targetModel.set_weights(mainModel.get_weights())

        bufferIndex += 1
        if bufferIndex >= bufferSize:
            bufferIndex = 0
        stepCount += 1
        targetCount += 1
        rewardSum += reward

        if (stepCount > 10000):
            break

    print("episode: {}  steps: {}  rewardSum: {}".format(i, stepCount, rewardSum))

    if stepCount > 2000:
        continueCount += 1
        if continueCount > 10:
            print('통과기준을 만족')
            break
    else:
        continueCount = 0

mainModel.save('model/avoidPoop.h5')


simulationThread.join()

