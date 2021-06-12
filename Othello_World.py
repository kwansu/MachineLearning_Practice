from Othello_Object import Cell
from time import sleep
import numpy as np
import pygame
import random
import tensorflow


class World_Othello:
    def __init__(self, sideLength,window, model: tensorflow.keras.Model) -> None:
        self.model = model
        self.window = window
        self.worldTime = 0.0
        self.stepInterval = 0.1
        self.cellLineCount = 8
        self.gameTurn = 0
        self.maxGameTurn = self.cellLineCount**2 - 4
        self.isBlackTurn = True
        self.width = sideLength
        self.height = sideLength
        self.cellSize = (sideLength/self.cellLineCount,
                         sideLength/self.cellLineCount)

        self.backGround = pygame.image.load(
            'python_simulation/backGround.png')
        self.sprite_white = pygame.image.load(
            'python_simulation/othello_stone_white.png')
        self.sprite_blakc = pygame.image.load(
            'python_simulation/othello_stone_black.png')

        self.cells = np.empty((self.cellLineCount**2), dtype=Cell)

        for x in range(0, self.cellLineCount):
            for y in range(0, self.cellLineCount):
                self.cells[x, y].pos = (x, y)
                self.cells[x, y].isEmpty = True

    def drawGrid(self):
        self.window.blit(self.backGround)
        for x in range(0, self.cellLineCount, self.cellSize[0]):
            pygame.draw.line(self.window, (0, 0, 0, 50), (x, 0), (x, self.height))
        for y in range(0, self.cellLineCount, self.cellSize[1]):
            pygame.draw.line(self.window, (0, 0, 0, 50), (0, y), (self.width, y))
    
    def drawCell(self,cell:Cell,isBlack):
        cell.isBlack = isBlack
        if isBlack:
            self.window.blit(self.sprite_blakc, cell.pos)
        else:
            self.window.blit(self.sprite_white, cell.pos)

    def setup(self):
        self.worldTime = 0
        self.gameTurn = 0

        for cell in self.cells:
            cell.isEmpty = True

        self.drawGrid(self.window)
        self.drawCell(self.window, self.cells[3, 3], False)
        self.drawCell(self.window, self.cells[4, 4], False)
        self.drawCell(self.window, self.cells[3, 4], True)
        self.drawCell(self.window, self.cells[4, 3], True)
    
    def put(self, pos, isBlack):
        cell : Cell = self.cells[pos[0],pos[1]]

        if cell.isEmpty == False:
            return 0

        cell.isEmpty = False
        self.drawCell(cell,isBlack)

        changedSum = 0
        
        for dir in range(0,8):
            changedSum += max(0,self.changeColor(cell.Around[dir],dir,isBlack))
        
        return changedSum

    
    def changeColor(self, cell:Cell, dir, isChangeToBlack):
        if cell == None or cell.isEmpty:
            return -1

        if cell.isBlack == isChangeToBlack:
            return 0
        
        result = self.changeColor(cell.Around[dir], dir, isChangeToBlack)
        if result >= 0:
            self.drawCell(cell,isChangeToBlack)
            return result+1
        
        return -1

    def step(self, action, state):
        state[action[0], action[1]] = 1 if self.isBlackTurn else 2
        changedSum = self.put(action, self.isBlackTurn)
        if changedSum <= 0:
            return -100, True

        self.isBlackTurn = ~self.isBlackTurn
        self.gameTurn += 1

        if self.gameTurn >= self.maxGameTurn:
            return 100+changedSum, True
        
        return changedSum, False
