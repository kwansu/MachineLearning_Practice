from Othello_Object import Stone
from time import sleep
import numpy as np
import pygame
import random
import tensorflow


class World_Othello:
    def __init__(self, sideLength, model: tensorflow.keras.Model) -> None:
        self.model = model
        self.isPlayig = True
        self.onStep = True
        self.worldTime = 0.0
        self.stepTime = 0.0
        self.stepInterval = 0.1
        self.width = 8
        self.height = 8
        self.sideLength = sideLength

        self.backGround = pygame.image.load('python_simulation/backGround_othello.png')
        self.stone_white = pygame.image.load('python_simulation/othello_stone_white.png')
        self.stone_blakc = pygame.image.load('python_simulation/othello_stone_black.png')

        self.objects = []
        #돌을 바닥갯수 만큼 만들어 놓고, 색도 놓는 순서에 맞게 지정해 놓으면 편할듯

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


    def step(self, action, state):
        reward = 1
        isTermimal = False

        for obj in self.objects:
            if obj.isActive == False:
                continue

        return reward, isTermimal
