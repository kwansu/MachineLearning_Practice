import numpy as np

class Cell:
    def __init__(self, pos=(-1,-1)) -> None:
        self.pos = np.array(pos)
        self.isEmpty = True
        self.isBlack = False
        self.bitAroundPutable = 0  #주변 8방향으로 둘 수 있는지를 비트 단위로 저장
        self.aroundCells = [None for i in range(8)]
    
    #현재 셀에서 주위 8방향에 대해 둘 수 있는 경우를 추가함
    #이미 추가되어있다면 false, 아니라면 true
    def addPutable(self, bitPutableInfo)->bool:
        if bitPutableInfo & self.bitAroundPutable == 0:
            self.bitAroundPutable+=bitPutableInfo
            return True
        
        return False

    #주위 8방향에 대해 더 이상 둘 수 있는 곳이 없으면 참, 아니면 거짓
    #정확하게는 가능리스트에서 삭제해야할 경우만 참
    def removePutable(self, bitPutableInfo) -> bool:
        if self.bitAroundPutable & bitPutableInfo == 0:
            return False
        
        self.bitAroundPutable -= bitPutableInfo
        return True if self.bitAroundPutable == 0 else False

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

    # UP_LEFT = 0b10
    # UP = 0b100
    # UP_RIGHT = 0b1000
    # LEFT = 0b10000
    # RIGHT = 0b100000
    # DOWN_LEFT = 0b1000000
    # DOWN_ = 0b10000000
    # DOWN_RIGHT = 0b100000000

    # PUT_IMPOSSIBLE = 0
    # PUT_CAN_BLACK = 1
    # PUT_CAN_WHITE = 2
    # PUT_OUT = 3

    # BIT_BLACK = 0b0001
    # BIT_WHITE = 0b0010
    # BIT_OUT = 0b0100
    # BIT_CHANGEABLE = 0b1000

    # BIT_DIR_MASK = 0b1111
    # BIT_OUT_MASK = 0b01000100010001000100010001000100
    # BIT_BLACK_MASK = 0b00010001000100010001000100010001
    # BIT_WHITE_MASK = 0b00100010001000100010001000100010


