from python_simulation.MainPyGame import MainPygame
from AvoidPoop_World import World_AvoidPoop as World
import tensorflow as tf
import numpy as np
import random
import threading


episodeCount = 5000
discountRate = 0.9

mainModel = tf.keras.models.Sequential()
mainModel = tf.keras.Sequential()
mainModel.add(tf.keras.layers.Dense(2883, tf.nn.relu, True, 'glorot_normal'))
mainModel.add(tf.keras.layers.Dense(961, tf.nn.relu, True, 'glorot_normal'))
#mainModel.add(tf.keras.layers.Dense(31, tf.nn.relu, True, 'glorot_normal'))
mainModel.add(tf.keras.layers.Dense(3, kernel_initializer='glorot_normal'))
mainModel.compile(tf.keras.optimizers.Adam(learning_rate=0.01), loss='mse')

targetModel = tf.keras.models.clone_model(mainModel)

reciprocalCount = 1.0/episodeCount
continueCount = 0  # 몇번 연속 기준을 통과했는지 판단용

width = 300
height = 400
world = World(width, height, targetModel)


def runSimulation():
    simulation = MainPygame(width=width, height=height, speed=1, fps=5)
    simulation.run(world)


simulationThread = threading.Thread(target=runSimulation)
simulationThread.start()

statesBuffer = np.zeros([2048, 961], dtype=int)
nextStatesBuffer = np.zeros([2048, 961], dtype = int)
actionsBuffer = np.zeros([2048], dtype=int)
rewardsBuffer = np.zeros([2048], dtype=int)
terminalsBuffer = np.zeros([2048], dtype=bool)


for i in range(episodeCount):
    e = i * reciprocalCount
    isTerminal = False
    sc = 0  # stepCount
    rewardSum = 0
    world.setupStepSimulation(statesBuffer[0])

    while not isTerminal:
        if random.random() > e:
            actionsBuffer[sc] = random.randrange(0, 3)
        else:
            x = np.reshape(statesBuffer[sc], [1, 961])
            actionsBuffer[sc] = np.argmax(mainModel.predict(x))

        rewardsBuffer[sc], isTerminal = world.step(
            actionsBuffer[sc], nextStatesBuffer[sc])
        
        rewardsBuffer[sc] *= sc
        
        terminalsBuffer[sc] = isTerminal
        statesBuffer[sc] = nextStatesBuffer[sc]
        sc += 1
        rewardSum += rewardsBuffer[sc]

        if (sc > 2000):
            break

    print("episode: {}  steps: {}  rewardSum: {}".format(i, sc, rewardSum))

    startStep = 0
    if sc > 14:
        startStep = 14

    Q_target = rewardsBuffer[startStep:sc] + discountRate * np.max(
        targetModel.predict(nextStatesBuffer[startStep:sc]), axis=1) * ~terminalsBuffer[startStep:sc]
    y = mainModel.predict(statesBuffer[startStep:sc])
    y[np.arange(sc-startStep), actionsBuffer[startStep:sc]] = Q_target

    mainModel.fit(statesBuffer[startStep:sc], y, sc, verbose=0)

    targetModel.set_weights(mainModel.get_weights())

    if sc > 2000:
        continueCount += 1
        if continueCount > 10:
            print('통과기준을 만족')
            break
    else:
        continueCount = 0

mainModel.save('model/avoidPoop.h5')

simulationThread.join()
