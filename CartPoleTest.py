import gym
import tensorflow
import numpy

environment = gym.make('CartPole-v0')
model = tensorflow.keras.models.load_model('model/cartpole2.h5')
state = environment.reset()
random_episodes = 0
rewardSum = 0

while random_episodes < 5:
    environment.render()

    x =model.predict(numpy.reshape(state, [1, 4]))
    action = numpy.argmax(x)
    nextState, reward, done, _ = environment.step(action)

    state = nextState
    rewardSum += reward

    if done:
        random_episodes += 1
        print("rewardSum", rewardSum)
        rewardSum = 0
        environment.reset()
