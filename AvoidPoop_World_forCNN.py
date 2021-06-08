from AvoidPoop_Objects import Player, Poop_Star, Object
from time import sleep
import numpy as np
import pygame
import random
import tensorflow


class World_AvoidPoop:
    def __init__(self, width, height, model: tensorflow.keras.Model) -> None:
        self.model = model
        self.isPlayig = True
        self.onStep = True
        self.worldTime = 0.0
        self.stepTime = 0.0
        self.stepInterval = 0.1
        self.poopInterval = 1.0
        self.width = 30
        self.height = 40
        self.playHeight = 30
        self.wRate = width/30
        self.hRate = height/40
        state = np.zeros([30, 30], dtype=int)
        self.action = 1

        self.playerSprite = pygame.image.load('python_simulation/player.png')
        self.poopSprite = pygame.image.load('python_simulation/poop.png')
        self.starSprite = pygame.image.load('python_simulation/star.png')
        self.backGround = pygame.image.load(
            'python_simulation/backGround_avoid.png')

        self.player = Player(sprite=self.playerSprite)
        self.player.pos = np.array([self.width/2, self.playHeight - 3])
        self.objects = []
        self.poolingPoops = []
        self.poolingStar = []

        for i in range(6):
            poop = Poop_Star(self.poopSprite)
            poop.isActive = False
            self.objects.append(poop)

        # self.setupStepSimulation()

    def checkCollision(self, player, object):
        d = abs(player.pos - object.pos) - (player.halfSize + object.halfSize)
        return d[0] < 0 and d[1] < 0

    def createObject(self):
        newObject = None
        if len(self.poolingPoops) > 0:
            newObject = self.poolingPoops.pop()
        else:
            newObject = Poop_Star(self.poopSprite)
            self.objects.append(newObject)

        newObject.isActive = True
        return newObject

    def update(self, window, deltaTime):
        window.blit(self.backGround, [0, 0])

        for obj in self.objects:
            if obj.isActive == False:
                continue

            window.blit(self.poopSprite, 10*(obj.pos - obj.halfSize))

        window.blit(self.playerSprite, 10 *
                    (self.player.pos - self.player.halfSize))

    def setupStepSimulation(self, state):
        self.worldTime = 0
        self.stepTime = 0

        for obj in self.objects:
            obj.isActive = False

        halfWidth = int(self.width/2)
        self.player.pos = np.array(
            [halfWidth, int(self.playHeight - self.player.halfSize[1]-1)])
        state.fill(0)

        #state[int(self.player.pos[0])*31 + int(self.player.pos[1])] = 1
        #self.player.updateState(state, self.width, self.playHeight)
        x = self.player.pos[0]-1
        y = self.player.pos[1]
        for h in range(y-2, y+3):
            state[h, x:x+3] = 1

        newObject = self.createObject()
        newObject.pos = [halfWidth, -1]

    def step(self, action, state):
        reward = 1
        isTermimal = False
        state.fill(0)

        self.player.pos = self.player.pos
        if action == 0:
            # if self.player.pos[0] > 0:
            self.player.pos[0] -= 1
        elif action == 2:
            # if self.player.pos[0] < self.width:
            self.player.pos[0] += 1

        if self.player.pos[0] <= 1 or self.player.pos[0] >= 28:
            isTermimal = True
            reward = -1

        #self.player.updateState(state, self.width, self.playHeight)
        x = self.player.pos[0]-1
        y = self.player.pos[1]
        for h in range(y-2, y+3):
            state[h, x:x+3] = 1

        self.worldTime += self.stepInterval
        if self.worldTime > self.poopInterval:
            self.worldTime = 0
            newObject = self.createObject()
            newObject.pos = [random.randrange(1, 28), -1]

        for obj in self.objects:
            if obj.isActive == False:
                continue

            obj.pos[1] += 1
            if obj.pos[1] >= self.playHeight:
                if obj.pos[1] >= self.height:
                    obj.isActive = False
                    self.poolingPoops.append(obj)
                continue
            else:
                state[int(obj.pos[1]), int(obj.pos[0])] = 1

            if self.checkCollision(self.player, obj):
                if obj.isStar == True:
                    reward += 1
                    pass
                else:
                    self.isPlayig = False
                    isTermimal = True
                    reward = 1

        return reward, isTermimal
