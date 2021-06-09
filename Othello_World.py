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
        self.cellLineCount = 8
        self.width = sideLength
        self.height = sideLength
        self.cellSize = (sideLength/self.cellLineCount,
                         sideLength/self.cellLineCount)
        self.gameTurn = 0

        self.backGround = pygame.image.load(
            'python_simulation/backGround_othello.png')
        self.sprite_white = pygame.image.load(
            'python_simulation/othello_stone_white.png')
        self.sprite_blakc = pygame.image.load(
            'python_simulation/othello_stone_black.png')

        self.stones: list(Stone) = np.array(
            (self.cellLineCount**2), dtype=Stone)

        for x in range(0, self.cellLineCount):
            for y in range(0, self.cellLineCount):
                self.stones[x, y].pos = (x, y)
                self.stones[x, y].isActive = False

    def drawGrid(self, window):
        window.blit(self.backGround)
        for x in range(0, self.cellLineCount, self.cellSize[0]):
            pygame.draw.line(window, (0, 0, 0, 50), (x, 0), (x, self.height))
        for y in range(0, self.cellLineCount, self.cellSize[1]):
            pygame.draw.line(window, (0, 0, 0, 50), (0, y), (self.width, y))

    def updateStone(self, window, stone : Stone, isBlack):
        stone.isActive =True
        stone.isBlack = isBlack
        if isBlack:
            window.blit(self.sprite_blakc, stone.pos)
        else:
            window.blit(self.sprite_white, stone.pos)

    def setup(self, window):
        self.worldTime = 0
        self.stepTime = 0
        self.gameTurn = 0

        for stone in self.stones:
            stone.isActive = False

        self.drawGrid(window)
        self.updateStone(window, self.stones[3, 3], False)
        self.updateStone(window, self.stones[4, 4], False)
        self.updateStone(window, self.stones[3, 4], True)
        self.updateStone(window, self.stones[4, 3], True)

    def step(self, action, state):
        reward = 1
        isTermimal = False

        for obj in self.objects:
            if obj.isActive == False:
                continue

        return reward, isTermimal
