from python_simulation.MainPyGame import MainPygame
from AvoidPoop_World import World_AvoidPoop as World
import tensorflow as tf
import numpy as np
import threading
import time

width = 300
height = 400

model = tf.keras.models.load_model('model/avoidPoop.h5')
world = World(width, height, model)


def runSimulation():
    simulation = MainPygame(width=width, height=height, speed=1, fps=5)
    simulation.run(world)

simulationThread = threading.Thread(target=runSimulation)
simulationThread.start()
stepSum = 0

for i in range(10):
    stepCount = 0
    state = np.zeros([961],dtype=int)
    world.setupStepSimulation(state)
    isTerminal = False

    while not isTerminal:
        time.sleep(0.03)

        actions = model.predict(np.reshape(state, [1, 961]))
        #print(actions)
        action = np.argmax(actions)

        reward, isTerminal = world.step(action,state)

        stepCount += 1
    
    stepSum += stepCount
    print(stepCount)

print(stepSum/10)

simulationThread.join()
