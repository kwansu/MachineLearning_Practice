import numpy as np

class Cell:
    UP_LEFT = 0
    UP = 1
    UP_RIGHT = 2
    LEFT = 3
    RIGHT = 4
    DOWN_LEFT = 5
    DOWN_ = 6
    DOWN_RIGHT = 7

    PUT_IMPOSSIBLE = 0
    PUT_CAN_BLACK = 1
    PUT_CAN_WHITE = 2
    PUT_OUT = 3

    BIT_BLACK = 0b0001
    BIT_WHITE = 0b0010
    BIT_OUT = 0b0100
    BIT_CHANGEABLE = 0b1000

    BIT_DIR_MASK = 0b1111
    BIT_OUT_MASK = 0b01000100010001000100010001000100
    BIT_BLACK_MASK = 0b00010001000100010001000100010001
    BIT_WHITE_MASK = 0b00100010001000100010001000100010



    def __init__(self, pos=(-1,-1)) -> None:
        self.pos = np.array(pos)
        self.isEmpty = True
        self.isBlack = False
        # 현재 셀을 중심으로 8방향에 대한 정보를 4비트씩나눠서 쓴다.
        # 비었는지, 블랙, 화이트, 변경가능(다른색으로 막혔을경우)인지 판단용
        self.bitAroundInfo = 0
        self.aroundCells = [None for i in range(8)]

    def setAroundCells(self, cells):
        self.__setAroundCells(self.pos[0]-1,self.pos[1]-1,cells,0)
        self.__setAroundCells(self.pos[0],self.pos[1]-1,cells,1)
        self.__setAroundCells(self.pos[0]+1,self.pos[1]-1,cells,2)
        self.__setAroundCells(self.pos[0]-1,self.pos[1],cells,3)
        self.__setAroundCells(self.pos[0]+1,self.pos[1],cells,4)
        self.__setAroundCells(self.pos[0]-1,self.pos[1]+1,cells,5)
        self.__setAroundCells(self.pos[0],self.pos[1]+1,cells,6)
        self.__setAroundCells(self.pos[0]+1,self.pos[1]+1,cells,7)

    def __setAroundCells(self, x, y, cells, dir):
        if x < 0 or x >= 8 or y < 0 or y >=8:
            return
        
        self.aroundCells[dir] = cells[x][y]

    def updateCellInfo(self, isBlack):
        self.isBlack = isBlack
        self.isEmpty = False

        pass
