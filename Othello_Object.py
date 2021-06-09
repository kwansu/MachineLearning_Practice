import numpy as np


class Stone:
    def __init__(self, pos = (-1,-1)) -> None:
        self.pos = np.array(pos)
        self.isEmpty = True
        self.canPut = False
        self.isBlack = True
