import gym
import random
import collections
import tensorflow as tf
import numpy

environment = gym.make('CartPole-v0')

state = environment.reset()
episodeCount = 500
discountRate = 0.99
batchSize = 64
targetInterval = 8  # 몇번마다 한번씩 메인을 타겟에 복사할지
rewardSum = 0
buffer = collections.deque(maxlen=10000)

inputSize = environment.observation_space.shape[0]

mainModel = tf.keras.models.Sequential()
mainModel = tf.keras.Sequential()
mainModel.add(tf.keras.layers.Dense(16, 'relu', True, 'glorot_normal'))
# 출력은 오른쪽,왼쪽 2개
mainModel.add(tf.keras.layers.Dense(2, kernel_initializer='glorot_normal'))
mainModel.compile(tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')

targetModel = tf.keras.models.clone_model(mainModel)

countR = 1 / episodeCount
continueCount = 0 # 몇번 연속 기준을 통과했는지 판단용
targetCount = targetInterval-batchSize

for i in range(episodeCount):
    e = countR * i
    isTerminal = False
    stepCount = 0
    state = environment.reset()

    while not isTerminal:
        if random.random() > e:
            action = environment.action_space.sample()
        else:
            x = numpy.reshape(state, [1, inputSize])
            action = numpy.argmax(mainModel.predict(x))

        nextState, reward, isTerminal, _ = environment.step(action)

        if isTerminal:
            reward = -1

        buffer.append((state, action, reward, nextState, isTerminal))

        if len(buffer) > batchSize:
            trainBatch = random.sample(buffer, batchSize)
            states = numpy.vstack([x[0] for x in trainBatch])
            actions = numpy.array([x[1] for x in trainBatch])
            rewards = numpy.array([x[2] for x in trainBatch])
            nextStates = numpy.vstack([x[3] for x in trainBatch])
            terminals = numpy.array([x[4] for x in trainBatch])

            Q_target = rewards + discountRate * numpy.max(targetModel.predict(nextStates), axis=1) * ~terminals

            y = mainModel.predict(states)
            y[numpy.arange(len(states)), actions] = Q_target

            mainModel.fit(states, y, batchSize, verbose=0)

        if targetCount > targetInterval:
            targetCount = 0
            targetModel.set_weights(mainModel.get_weights())

        state = nextState
        stepCount += 1
        targetCount += 1

    print("episode: {}  steps: {}".format(i, stepCount))

    if stepCount > 150:
        continueCount += 1
        if continueCount > 100:
            print('통과기준을 만족')
            break
    else:
        continueCount = 0

mainModel.save('model/cartpole.h5')

episodeCount = 0
while episodeCount < 10:
    environment.render()
    action = random.randrange(0, 2)
    nextState, reward, done, _ = environment.step(action)

    state = nextState
    rewardSum += reward

    if done:
        episodeCount += 1
        print("rewardSum", rewardSum)
        rewardSum = 0
        environment.reset()
