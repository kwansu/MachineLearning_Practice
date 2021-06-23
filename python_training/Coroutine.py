def calc():
    result = 0
    while True:
        str = (yield result)
        a, operator, b = str.split()
        a = int(a)
        b = int(b)
        if operator == '+':
            result = a + b
        elif operator == '-':
            result = a - b
        elif operator == '*':
            result = a * b
        elif operator == '/':
            result = a / b

expressions = ("11 + 5", "3 * -2", "11 / 2")
 
c = calc()
next(c)
 
for e in expressions:
    print(c.send(e))
 
c.close()