from python_simulation.MainPyGame import MainPygame
from Othello_World import World_Othello as World
import tensorflow as tf
import numpy as np
import collections
import random
import threading


episodeCount = 1000
discountRate = 0.99

mainModel = tf.keras.Sequential()
mainModel.add(tf.keras.layers.Conv2D(16, kernel_size=(5, 5), padding='SAME',
              activation='relu', kernel_initializer='glorot_normal'))
mainModel.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), padding='SAME',
              activation='relu', kernel_initializer='glorot_normal'))
mainModel.add(tf.keras.layers.Flatten())
mainModel.add(tf.keras.layers.Dense(256, tf.nn.relu, True, 'glorot_normal'))
mainModel.add(tf.keras.layers.Dense(64, kernel_initializer='glorot_normal'))
mainModel.compile(tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
targetModel = tf.keras.models.clone_model(mainModel)

bufferSize = 10000
buffer = collections.deque(maxlen=bufferSize)

statesBuffer = np.zeros([bufferSize+1, 8, 8, 1])
batchSize = 64
targetInterval = 10
targetCount = targetInterval-batchSize

lineLenth = 320
simulation = MainPygame(lineLenth, lineLenth, speed=1, fps=5)
world = World(lineLenth,simulation.window, targetModel)

def runSimulation():
    simulation.run(world)


simulationThread = threading.Thread(target=runSimulation)
simulationThread.start()

stateIndex = 0

for i in range(episodeCount):
    e = 1. / ((i / 5) + 1)
    isTerminal = False
    stepCount = 0
    rewardSum = 0
    world.setup(statesBuffer[stateIndex])

    while not isTerminal:
        state = statesBuffer[stateIndex]
        nextState = statesBuffer[stateIndex+1]

        if random.random() < e:
            action = random.randrange(0, 64)
        else:
            x = np.reshape(state, [1, 8, 8, 1])
            action = np.argmax(mainModel.predict(x))

        reward, isTerminal = world.step(action, nextState)

        buffer.append([state, action, reward, nextState, isTerminal])

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

        #state = nextState
        stateIndex += 1
        if stateIndex >= bufferSize:
            stateIndex = 0

        stepCount += 1
        targetCount += 1
        rewardSum += reward

        if (stepCount > 2000):
            break

    print("episode: {}  steps: {}  rewardSum: {}".format(i, stepCount, rewardSum))


mainModel.save('model/othello.h5')
simulationThread.join()
