import numpy as np

class AAA:
    def __init__(self, col, row) -> None:
        self.pos = (col,row)


l = tuple(tuple(AAA(col,row) for col in range(8)) for row in range(8))


for cols in l:
    print()
    for row in cols:
        print(row.pos,end=',')

print(l[3][4].pos)

        