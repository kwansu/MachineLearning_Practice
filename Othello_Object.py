import numpy as np


class Object:
    def __init__(self, pos=[0, 0], isActive=True) -> None:
        self.pos = np.array(pos)
        self.isActive = isActive


class Stone(Object):
    def __init__(self, pos) -> None:
        super().__init__(pos)
        isBlack = True
