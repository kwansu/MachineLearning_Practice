col, row = map(int,input().split())
matrix = []
for line in range(row):
    matrix.append(list(input()))
for i in range(row):
    for j in range(col):
        if matrix[i][j] == '.':
            matrix[i][j] = 0
            for y in range(max(0,i-1), min(row,i+2)):
                for x in range(max(0,j-1), min(col,j+2)):
                    if (x != j or y != i) and matrix[y][x] == '*':
                        matrix[i][j] += 1
for line in matrix:
    for value in line:
        print(value, end='')
    print()