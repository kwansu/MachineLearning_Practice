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

    def __init__(self, pos=(-1,-1)) -> None:
        self.pos = np.array(pos)
        self.isEmpty = True
        self.isBlack = False
        self.dir = np.zeros(8,dtype=int)
