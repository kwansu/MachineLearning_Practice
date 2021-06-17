from Othello_Object import Cell
from time import sleep
import numpy as np
import pygame
import random
import tensorflow


class World_Othello:
    def __init__(self, sideLength, window, model: tensorflow.keras.Model) -> None:
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
        self.cellSize = (int(sideLength/self.cellLineCount),
                         int(sideLength/self.cellLineCount))

        #self.putableList = []
        self.blackPutableList = []
        self.whitePutableList = []

        self.backGround = pygame.image.load(
            'python_simulation/backGround.png')
        self.sprite_white = pygame.image.load(
            'python_simulation/othello_stone_white.png')
        self.sprite_blakc = pygame.image.load(
            'python_simulation/othello_stone_black.png')
        self.temp = pygame.image.load(
            'python_simulation/othello_temp.png')
        self.temp2 = pygame.image.load(
            'python_simulation/othello_temp2.png')

        self.cells = tuple(tuple(Cell((col, row)) for row in range(
            self.cellLineCount)) for col in range(self.cellLineCount))

        for colums in self.cells:
            for cell in colums:
                cell.setAroundCells(self.cells)

    def drawGrid(self):
        self.window.blit(self.backGround, (0,0))
        for x in range(0, self.width, self.cellSize[0]):
            pygame.draw.line(self.window, (0, 0, 0, 50),
                             (x, 0), (x, self.height))
        for y in range(0, self.height, self.cellSize[1]):
            pygame.draw.line(self.window, (0, 0, 0, 50),
                             (0, y), (self.width, y))

    def drawCell(self, cell: Cell, isBlack):
        cell.isBlack = isBlack
        if isBlack:
            self.window.blit(self.sprite_blakc, cell.pos * self.cellSize)
        else:
            self.window.blit(self.sprite_white, cell.pos* self.cellSize)

    def setup(self, state = None):
        self.worldTime = 0
        self.gameTurn = 0
        self.isBlackTurn = True
        self.blackPutableList.clear()
        self.whitePutableList.clear()

        for cols in self.cells:
            for cell in cols:
                cell.isEmpty = True
                cell.bitAroundPutableBlack = 0
                cell.bitAroundPutableWhite = 0

        self.drawGrid()
        self.put((3,3), False)
        self.put((4,4), False)
        self.put((3,4), True)
        self.put((4,3), True)

        if state != None:
            state.fill(0)
            state[3,3] = 1
            state[4,4] = 1
            state[3,4] = -1
            state[4,3] = -1

    def put(self, pos, isBlack):
        self.checkPutable(True)
        self.checkPutable(False)

        cell:Cell = self.cells[pos[0]][pos[1]]

        if cell.isEmpty == False:
            return 0

        cell.isEmpty = False
        self.drawCell(cell, isBlack)
        bitAroundPutable = cell.bitAroundPutableBlack if isBlack else cell.bitAroundPutableWhite
        putableList = self.blackPutableList if isBlack else self.whitePutableList

        changedSum = 0

        for dir in range(0, 8):
            if cell.getBitAroundPutable() & 1<<dir != 0:
                changedSum += self.changeColor(cell.aroundCells[dir], dir, isBlack)
        
        cell.bitAroundPutableBlack = 0
        cell.bitAroundPutableWhite = 0
        self.updatePutableList(cell)
        self.removePutableList(cell)

        ####체크용#######################
        self.checkPutable(True)
        self.checkPutable(False)
        
        if changedSum == 0:
            aa = 0
        ####################################

        return changedSum

    def addPutableDirection(self, cell :Cell, dir):
        dirR = 7-dir
        if cell.getBitAroundPutable() &1<<dirR == 0:
            if cell.addDirectionPutable(dirR,cell.isBlack):
                self.removePutableListColor(cell,not cell.isBlack)
            nextCell:Cell = cell.aroundCells[dir]
            if nextCell == None:
                return
            if nextCell.isEmpty:
                if nextCell.addDirectionPutable(dirR,not cell.isBlack):
                    self.removePutableListColor(nextCell,cell.isBlack)
                self.addPutableList(nextCell,not cell.isBlack)
            elif nextCell.isBlack == cell.isBlack:
                self.addPutableDirection(nextCell,dir)

    def removePutableDirection(self, cell:Cell, dir):
        dirR = 7-dir
        if cell.getBitAroundPutable() &1<<dirR != 0:
            cell.removeDirectionPutable(dirR)
            nextCell:Cell = cell.aroundCells[dir]
            if nextCell == None:
                return
            if nextCell.isEmpty:
                nextCell.removeDirectionPutable(dirR)
                self.removePutableListColor(nextCell,not cell.isBlack)
            elif nextCell.isBlack == cell.isBlack:
                self.removePutableDirection(nextCell,dir)

    def updatePutableList(self, cell :Cell):
        for dir in range(8):
            dirR = 7-dir
            nextCell :Cell= cell.aroundCells[dir]
            if nextCell == None:
                return
            if nextCell.isEmpty:
                if cell.getBitAroundPutable() & 1<<dirR != 0:
                    if nextCell.addDirectionPutable(dirR,not cell.isBlack):
                        self.removePutableListColor(nextCell,cell.isBlack)
                    self.addPutableList(nextCell,not cell.isBlack)
                    if nextCell.getBitAroundPutableColor(cell.isBlack) == 0:
                        self.removePutableListColor(nextCell,cell.isBlack)
                else:
                    nextCell.removeDirectionPutable(dirR)
                    self.removePutableListColor(nextCell,cell.isBlack)
            elif nextCell.isBlack == cell.isBlack:
                if nextCell.getBitAroundPutable() & 1<<dir != 0:
                    self.addPutableDirection(cell,dirR)
                if cell.getBitAroundPutable() & 1<<dirR == 0:
                    self.removePutableDirection(nextCell,dir)
            else:
                self.addPutableDirection(nextCell,dir)
                self.addPutableDirection(cell,dirR)
    
    def checkPutableDirection(self, cell : Cell, dir, isBlack):
        if cell == None or cell.isEmpty:
            return False
        
        if cell.isBlack == cell.isBlack:
            return True

        return self.checkPutableDirection(cell.aroundCells[dir],dir,isBlack)

    def sibal(self, cell:Cell,isBlack):
        isPutable = False
        for dir in range(8):
            nextCell :Cell= cell.aroundCells[dir]
            if nextCell == None or nextCell.isEmpty or nextCell.isBlack == isBlack:
                continue
            isPutable = self.checkPutableDirection(nextCell,dir,isBlack)
            if isPutable:
                break
            
        if self.gameTurn != 0:
            if isPutable == False:
                a = 0

    def checkPutable(self, isBlack):
        putableList = self.blackPutableList if isBlack else self.whitePutableList
        for cell in putableList:
            self.sibal(cell,isBlack)
    
    def addPutableList(self, cell:Cell, isBlack):
        putableList = self.blackPutableList if isBlack else self.whitePutableList
        if cell in putableList:
            return

        self.sibal(cell,isBlack)
        putableList.append(cell)
        if cell.isEmpty and not isBlack:
            self.window.blit(self.temp, cell.pos* self.cellSize)
    
    def removePutableListColor(self, cell:Cell, isBlack):
        if cell.getBitAroundPutableColor(not isBlack) != 0:
            return
        putableList = self.blackPutableList if isBlack else self.whitePutableList
        if cell in putableList:
            putableList.remove(cell)
        if not isBlack and cell.isEmpty:
            self.window.blit(self.temp2, cell.pos* self.cellSize)

    def removePutableList(self, cell:Cell):
        if cell in self.blackPutableList:
            self.blackPutableList.remove(cell)
        if cell in self.whitePutableList:
            self.whitePutableList.remove(cell)

    def changeColor(self, cell: Cell, dir, isChangeToBlack):
        #cell.changeColor(isChangeToBlack)
        assert(cell.isBlack != isChangeToBlack)
        cell.bitAroundPutableBlack = 0
        cell.bitAroundPutableWhite = 0
        self.drawCell(cell, isChangeToBlack)
        self.updatePutableList(cell)
        
        nextCell :Cell= cell.aroundCells[dir]
        assert(nextCell.isEmpty == False)

        if nextCell.isBlack != isChangeToBlack:
            return self.changeColor(nextCell,dir,isChangeToBlack) +1
        
        return 1

    def step(self, action, state):
        state[action[0], action[1]] = 1 if self.isBlackTurn else 2
        changedSum = self.put(action, self.isBlackTurn)
        if changedSum <= 0:
            return -100, True

        self.isBlackTurn = not self.isBlackTurn
        self.gameTurn += 1

        if self.gameTurn >= self.maxGameTurn:
            return 100+changedSum, True

        return changedSum, False

    def randomPut(self):
        self.gameTurn+=1

        putableList = self.blackPutableList if self.isBlackTurn else self.whitePutableList
        
        if len(putableList) == 0:
            self.setup()
            return
        randCell = random.choice(putableList)
        changedSum = self.put(randCell.pos, self.isBlackTurn)
        if changedSum <= 0:
            self.setup()
            return
        
        self.isBlackTurn = not self.isBlackTurn
    
    def update(self, putPos):
        self.gameTurn+=1

        if putPos == None:
            return

        if putPos[0] < 0 or putPos[1] < 0 or putPos[0] > self.width or putPos[1] > self.height:
            return
        
        putPos[0] = int(putPos[0]/self.cellSize[0])
        putPos[1] = int(putPos[1]/self.cellSize[1])

        changedSum = self.put(putPos, self.isBlackTurn)
        if changedSum <= 0:
            self.setup()
            return

        self.isBlackTurn = not self.isBlackTurn

        if self.gameTurn >= self.maxGameTurn:
            self.setup()
            return
