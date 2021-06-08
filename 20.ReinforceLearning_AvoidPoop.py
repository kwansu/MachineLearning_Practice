from python_simulation.MainPyGame import MainPygame
from AvoidPoop_World import World_AvoidPoop as World
import tensorflow as tf
import numpy as np
import collections
import random
import threading
import time


episodeCount = 2000
discountRate = 0.9
batchSize = 64
targetInterval = 10
buffer = collections.deque(maxlen=20000)

mainModel = tf.keras.models.Sequential()
mainModel = tf.keras.Sequential()
mainModel.add(tf.keras.layers.Dense(256, 'relu', True, 'glorot_normal'))
mainModel.add(tf.keras.layers.Dense(128, 'relu', True, 'glorot_normal'))
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

for i in range(episodeCount):
    e = 1. / ((i / 10) + 1)
    #e = countR * i
    isTerminal = False
    stepCount = 0
    rewardSum = 0
    state = world.setupStepSimulation()

    while not isTerminal:
        if random.random() < e:
            action = random.randrange(0, 2)
        else:
            x = np.reshape(state, [1, 961])
            action = np.argmax(mainModel.predict(x))

        nextState = None
        reward, isTerminal = world.step(action, nextState)

        buffer.append([state, action, reward, nextState, isTerminal])

        if len(buffer) > batchSize:
            trainBatch = random.sample(buffer, batchSize)
            states = np.vstack([x[0] for x in trainBatch])
            actions = np.array([x[1] for x in trainBatch])
            rewards = np.array([x[2] for x in trainBatch])
            nextStates = np.vstack([x[3] for x in trainBatch])
            terminals = np.array([x[4] for x in trainBatch])

            Q_target = rewards + discountRate * \
                np.max(targetModel.predict(nextStates), axis=1) * ~terminals

            y = mainModel.predict(states)
            y[np.arange(64), actions] = Q_target

            mainModel.fit(states, y, batchSize, verbose=0)

        if targetCount > targetInterval:
            targetCount = 0
            targetModel.set_weights(mainModel.get_weights())

        state = nextState
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

world = World(width, height, mainModel)

for i in range(5):
    stepCount = 0
    state = world.setupStepSimulation()
    isTerminal = False

    while not isTerminal:
        time.sleep(0.01)

        x = np.reshape(state, [1, 961])
        action = np.argmax(mainModel.predict(x))

        nextState, reward, isTerminal = world.step(action)

        state = nextState
        stepCount += 1
    
    print(stepCount)

simulationThread.join()

