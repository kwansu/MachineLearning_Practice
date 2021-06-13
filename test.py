import numpy as np
from numpy.core.numeric import count_nonzero

class AAA:
    count = 0

    def __init__(self, num) -> None:
        self.num = num +10
        AAA.count +=1


l = [AAA(i) for i in range(10)]

for aaa in l:
    print(aaa.num)