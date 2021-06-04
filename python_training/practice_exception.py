class SoldOutError(Exception):
    def __str__(self) -> str:
        return "재고가 소진되어 더 이상 주문을 받지 않습니다."

chicken = 10
waiting = 1 # 홀 안에는 만석. 대기번호는 1부터 시작

try:
    while(1):
        print("[남은 치킨 : {}]".format(chicken))
        order = int(input("치킨 몇마리 주문 하셨어요?"))
        if type(order) != int or order < 1:
            raise ValueError
        elif order > chicken:
            print("재료가 부족합니다.")
        else:
            print("[대기번호 {0}] {1} 마리 주문이 완료되었습니다.".format(waiting,order))
            waiting += 1
            chicken -= order

        if chicken <= 0:
            raise SoldOutError

except ValueError:
    print("잘못된 값을 입력하셨습니다.")
except SoldOutError as err:
    print(err)



