import pygame
from World import World

'''
위로 방향키     :   배속증가
아래 방향키     :   배속감소
스페이스        :   일시정지/해제

유전 알고리즘에 따른 자연선택 => 현재 환경에서는 처음에는 속도가 들쭉날쭉 생성되지만,
일정 수준이상의 개체가 되면 서서히 빠른 개체만 살아남고, 수십번의 세대교체후에는
대부분 속도 250 이상의 개체만 살아남게된다.
순간 회전각과 회전지속 시간들을 유전 변수로 주자 속도 전환이 매우 느려졌다.
아마 많이 회전하면서 놓친 음식을 먹는 개체들이 오래 살아서 그런것 같지만,
결국 종래에 빠른 개체들에게 밀려서 전부 빠른개체만 살아남았다.
'''

pygame.init()

width = 500
height = 500

window = pygame.display.set_mode((width, height))
backGround = pygame.image.load('backGround.png')

world = World(width, height)
world.setup(12, 40)
clock = pygame.time.Clock()

# 재생속도를 너무 높일 경우 이동,회전,충돌 물리계산에 문제가 생길 수 있다.
timeSpeed = 0.004 # 1배속 = 0.001

isRunning = True
isStoped = False
while isRunning:
    deltaTime = clock.tick()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            isRunning = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                isStoped = not isStoped
            if event.key == pygame.K_UP:
                timeSpeed *= 2.0 #2배씩 빨라짐
                print('재생속도 :', timeSpeed*1000)
            if event.key == pygame.K_DOWN:
                print('재생속도 :', timeSpeed*1000)
                timeSpeed *= 0.5 #0.5배
    
    if isStoped:
        continue
    window.blit(backGround, [0, 0])
    world.updateCollision()
    world.update(window, deltaTime*timeSpeed)
    pygame.display.update()

pygame.quit()
