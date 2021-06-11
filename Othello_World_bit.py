from Othello_Object import Cell
from time import sleep
import numpy as np
import pygame
import random
import tensorflow


class World_Othello:
    UP_LEFT = (-1,-1)
    UP = (0,-1)
    UP_RIGHT = (-1,1)
    LEFT = (-1,0)
    RIGHT = (1,0)
    DOWN_LEFT = (-1,1)
    DOWN = (0,1)
    DOWN_RIGHT = (1,1)

    DIRECTION = (UP_LEFT,UP,UP_RIGHT,LEFT,RIGHT,DOWN_LEFT,DOWN,DOWN_RIGHT)

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

        self.cells = np.array((self.cellLineCount**2), dtype=Cell)

        for x in range(0, self.cellLineCount):
            for y in range(0, self.cellLineCount):
                self.cells[x, y].pos = (x, y)
                self.cells[x, y].isEmpty = True

    def drawGrid(self, window):
        window.blit(self.backGround)
        for x in range(0, self.cellLineCount, self.cellSize[0]):
            pygame.draw.line(window, (0, 0, 0, 50), (x, 0), (x, self.height))
        for y in range(0, self.cellLineCount, self.cellSize[1]):
            pygame.draw.line(window, (0, 0, 0, 50), (0, y), (self.width, y))
    
    def drawCell(self,window,cell:Cell,isBlack):
        cell.isBlack = isBlack
        if isBlack:
            window.blit(self.sprite_blakc, cell.pos)
        else:
            window.blit(self.sprite_white, cell.pos)

    def setup(self, window):
        self.worldTime = 0
        self.stepTime = 0
        self.gameTurn = 0

        for cell in self.cells:
            cell.isEmpty = False

        self.drawGrid(window)
        self.drawCell(window, self.cells[3, 3], False)
        self.drawCell(window, self.cells[4, 4], False)
        self.drawCell(window, self.cells[3, 4], True)
        self.drawCell(window, self.cells[4, 3], True)

    def changeCell(self, cell : Cell, isBlack):
        assert cell.isBlack != isBlack
        cell.isBlack = isBlack
        bitColor = Cell.BIT_BLACK if isBlack else Cell.BIT_WHITE
        bitCheck = bitColor | Cell.BIT_OUT

        col = cell.pos[0]
        row = cell.pos[1]

        # 모든 방향(4비트씩)에 대한 정보 업데이트
        for i in range(0,8):
            #현재 방향에 대한 비트정보만 남긴다.
            bit = cell.bitAroundInfo
            bitDirInfo = cell.bitAroundInfo & (Cell.BIT_DIR_MASK << i)
            #현재 방향에 대해 out(범위밖)이라면 넘긴다.
            if Cell.BIT_OUT_MASK & bitDirInfo != 0:
                continue

            bitReverseDir = Cell.BIT_DIR_MASK << (7-i)
            bitDirCheck = bitCheck << i
    
    def updateCell(self, row, col, dir,bitDirCheck, bitReverseInfo,bitReverseDir,isFirst):
        col+=dir[0]
        row+=dir[1]
        cell : Cell = self.cells[col,row]
        bit = cell.bitAroundInfo

        #정보가 이미 같다면 리턴
        if (bit & bitReverseDir) == bitReverseInfo:
            return

        #셀에서 부른(call) 방향에 대한 정보를 갱신한다.
        cell.bitAroundInfo = (bit & (~bitReverseDir)) | bitReverseInfo

        #계속 같은 방향으로 색이 변하지 않는다면 재귀호출(막혀있는지도 검사)
        if bitDirCheck & bit == 0:
            if isFirst:
                if cell.isBlack:
                    #bitReverseInfo = 
                    pass
                else:
                    pass

            self.updateCell(row,col,dir,bitDirCheck,bitReverseInfo,bitReverseDir,False)
    
    def checkColor(self, row, col, isBlack):
        pass
        

    def changeColor(self, row, col, direction, isBlack):
        row+=direction[0]
        col+=direction[1]
        if row < 0 or row >= self.cellLineCount or col < 0 or col >= self.cellLineCount:
            return self.OUT
        
        return self.CAN_PUT

    def step(self, action, state):
        reward = 1
        isTermimal = False

        for obj in self.objects:
            if obj.isActive == False:
                continue

        return reward, isTermimal
