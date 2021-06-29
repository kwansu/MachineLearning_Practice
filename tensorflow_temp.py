class Student:
    def __init__(self, name, height):
        self.name = name
        self.height = height


any_class = [Student('박이김', 160), Student('외계인', 210),
            Student('강먼저', 180), Student('홍길동', 160)]


any_class.sort(key=lambda x: x.name)
# ('강먼저', 180), ('박이김', 160), ('외계인', 210), ('홍길동', 160)

any_class.sort(key=lambda x: (x.height, x.name))
# ('박이김', 160), ('홍길동', 160), ('강먼저', 180), ('외계인', 210)

any_class.sort(key=lambda x: (-x.height, x.name))
# ('외계인', 210), ('강먼저', 180), ('박이김', 160), ('홍길동', 160)

any_class.sort(key=lambda x: x.name, reverse=True)
any_class.sort(key=lambda x: x.height)




print(any_class[0].name)
print(any_class[1].name)
print(any_class[2].name)
print(any_class[3].name)