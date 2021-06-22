def countdown(n):
    count = n+1
    def down():
        nonlocal count
        count -= 1
        return count
    return down

n = 10
 
c = countdown(n)
for i in range(n):
    print(c(), end=' ')