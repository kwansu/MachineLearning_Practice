import numpy as np


class Object:
    def __init__(self, sprite, pos=[0, 0], isActive=True) -> None:
        self.pos = np.array(pos)
        self.halfSize = np.array(sprite.get_size()) * 0.05
        self.isActive = isActive
    
    def updateState(self, state, width, height):
        start = self.pos - self.halfSize
        end = self.pos + self.halfSize
        # start[0] = max(0,start[0])
        # start[1] = max(0,start[1])
        # end[0] = min(width, end[0])
        # end[1] = min(height, end[1])
        
        for i in range(int(start[1]+1), int(end[1])):
            hi = i*width
            state[hi + int(start[0]) + 1: hi+int(end[0])] = 1


class Poop_Star(Object):
    def __init__(self, sprite) -> None:
        super().__init__(sprite=sprite)
        self.isStar = False

    def update(self, world, deltaTime):
        self.pos[1] += 10 * deltaTime
        if self.pos[1] > world.height:
            self.isActive = False
            world.poolingPoops.append(self)
        
        return self.pos


class Player(Object):
    def __init__(self, sprite) -> None:
        super().__init__(sprite=sprite)
        self.score = 0.0
