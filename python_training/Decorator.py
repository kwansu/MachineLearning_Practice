def decorator1(func):
    def aa():
        func()
        print('decor1')
    return aa
def decorator2(func):
    def aa():
        func()
        print('decor2')
    return aa

@decorator1
@decorator2
def func():
    print("func")

func()