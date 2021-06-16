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
        self.putableList.clear()

        for cols in self.cells:
            for cell in cols:
                cell.isEmpty = True
                cell.bitAroundPutable = 0

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
        cell:Cell = self.cells[pos[0]][pos[1]]

        if cell.isEmpty == False:
            return 0

        cell.isEmpty = False
        self.drawCell(cell, isBlack)
        if cell.bitAroundPutable != 0:
            cell.bitAroundPutable = 0
            self.putableList.remove(cell)

        changedSum = 0

        for dir in range(0, 8):
            if (isBlack and cell.bitAroundPutableBlack & 1<<dir != 0) or (not isBlack and cell.bitAroundPutableWhite & 1<<dir != 0):
                changedSum += max(0,self.changeColor(cell.aroundCells[dir], dir, isBlack))
        
        self.updatePutableList(cell,isBlack)
        return changedSum

    #인접방향에 대한 정보. 같은색이면 계속 재귀해서 확인한다.
    DIR_INFO_EMPTY = -2
    DIR_INFO_NONE = -1
    DIR_INFO_BLOCK = 0

    def checkCellDirectionInfo(self,cell:Cell,dir,isBlack):
        if cell == None:
            return World_Othello.DIR_INFO_NONE, None
        if cell.isEmpty:
            return World_Othello.DIR_INFO_EMPTY, cell
        if cell.isBlack != isBlack:
            return World_Othello.DIR_INFO_BLOCK, None

        return self.checkCellDirectionInfo(cell.aroundCells[dir],dir,isBlack)

    def updatePutableList(self, cell :Cell):
        for dir in range(8):
            dirR = 7-dir
            nextCell :Cell= cell.aroundCells[dir]
            if nextCell == None:
                return
            if nextCell.isEmpty:
                if cell.getBitAroundPutable() & dirR != 0:
                    nextCell.addDirectionPutable(dirR,not cell.isBlack)
                    self.addPutableList(nextCell,)
                if nextCell.getBitAroundPutableColor(cell.isBlack) == 0:
                    self.removePutableList(nextCell)
            elif nextCell.isBlack == cell.isBlack:
                if nextCell.getBitAroundPutable() & dir != 0:
                    pass
                if cell.getBitAroundPutable() & dirR == 0:
                    pass
            else:
                pass

            # dirInfo, emptyCell = self.checkCellDirectionInfo(cell.aroundCells[dir],dir,isBlack)
            # dirInfoR, emptyCellR = self.checkCellDirectionInfo(cell.aroundCells[dirR],dirR,isBlack)

            # if dirInfo == World_Othello.DIR_INFO_EMPTY:
            #     if dirInfoR == World_Othello.DIR_INFO_BLOCK:
            #         self.addPutableList(emptyCell,dir)
            #     else:
            #         self.removePutableList(emptyCell,dir)
            
            # if dirInfoR == World_Othello.DIR_INFO_EMPTY:
            #     if dirInfo == World_Othello.DIR_INFO_BLOCK:
            #         self.addPutableList(emptyCellR,dirR)
            #     else:
            #         self.removePutableList(emptyCellR,dirR)
    
    def addPutableList(self, cell:Cell, isBlack):
        putableList = self.blackPutableList if isBlack else self.whitePutableList
        if cell in putableList:
            return
        putableList.append(cell)
        self.window.blit(self.temp, cell.pos* self.cellSize)
    
    def removePutableList(self, cell:Cell):
        self.blackPutableList.remove(cell)
        self.whitePutableList.remove(cell)
        if cell.isEmpty:
            self.window.blit(self.temp2, cell.pos* self.cellSize)
    
    def changeColor(self, cell: Cell, dir, isChangeToBlack):
        cell.changeColor(isChangeToBlack)
        self.updatePutableList(cell)
        
        if cell.isEmpty:
            return World_Othello.DIR_INFO_EMPTY
        if cell.isBlack == isChangeToBlack:
            return World_Othello.DIR_INFO_BLOCK

        result = self.changeColor(cell.aroundCells[dir], dir, isChangeToBlack)
        if result >= 0:
            self.drawCell(cell, isChangeToBlack)
            self.updatePutableList(cell,isChangeToBlack)
            return result+1

        return World_Othello.DIR_INFO_NONE

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
    
    def update(self, putPos):
        if putPos == None:
            return

        if putPos[0] < 0 or putPos[1] < 0 or putPos[0] > self.width or putPos[1] > self.height:
            return
        
        putPos[0] = int(putPos[0]/self.cellSize[0])
        putPos[1] = int(putPos[1]/self.cellSize[1])

        changedSum = self.put(putPos, True)
        if changedSum <= 0:
            self.setup()
            return

        #self.isBlackTurn = not self.isBlackTurn
        randCell = random.choice(self.putableList)
        changedSum = self.put(randCell.pos, False)
        if changedSum <= 0:
            self.setup()
            return

        if self.gameTurn >= self.maxGameTurn:
            self.setup()
            return
