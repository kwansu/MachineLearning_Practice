from Othello_World import World_Othello as World
import pygame
import numpy as np
import tensorflow as tf

pygame.init()
width = 320
height = 320
window = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()
timeSpeed = 0.001 * 1 # 1배속 = 0.001
minDeltaTime = 0.5 * 1000
fps = 60
isRunning = True
isStoped = False

model :tf.keras.Model = tf.keras.models.load_model('model/othello.h5')
state = np.zeros([8,8],dtype=int)
world = World(320, window)
world.setup(state)

while isRunning:
    deltaTime = clock.tick(fps)
    if deltaTime > minDeltaTime:
        deltaTime = minDeltaTime
    
    if world.isBlackTurn:
        x = np.reshape(state, [1, 8, 8, 1])
        action = np.argmax(model.predict(x))
        actionPos = (action&0b111,action>>3)
        world.putCell(actionPos)
    else:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                isRunning = False
            if event.type == pygame.KEYDOWN:
                # if event.key == pygame.K_SPACE:
                #     world.setup()
                if event.key == pygame.K_SPACE:
                    world.randomPut()

            if event.type == pygame.MOUSEBUTTONUP:
                world.putPlayer(list(event.pos))

    if isStoped:
        continue
    
    pygame.display.update()
pygame.quit()     