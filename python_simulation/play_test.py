from math import *
from bisect import bisect_left
import pygame
import random
import numpy as np


class Object:
    def __init__(self, pos, radius) -> None:
        self.pos = np.array(pos)
        self.radius = radius
        self.isActive = True

    def activate(self, isActive):
        self.isActive = isActive

    def update(self, world, deltaTime):
        pass


class Food(Object):
    def __init__(self, sprite, pos=[0., 0.], radius=10.) -> None:
        super().__init__(pos = pos, radius = radius)
        self.sprite = sprite


class Agent(Object):
    def __init__(self, sprite, pos=[0., 0.], radius=15.) -> None:
        super().__init__(pos, radius)
        self.spd = 50.
        self.dir = np.array([1, 0])  # 항상 normalize 시킬것
        self.rotate(random.randrange(0, 360))
        self.interval = 0.
        self.degree = 0.
        self.sprite = sprite

    def rotate(self, degree):
        r = radians(degree)
        mat = np.array([[cos(r), -sin(r)], [sin(r), cos(r)]])
        self.dir = np.dot(self.dir, mat)
        self.dir /= np.linalg.norm(self.dir)

    def lookAt(self, atPos):
        self.dir = atPos - self.pos
        self.dir /= np.linalg.norm(self.dir)

    def update(self, world, deltaTime):
        # self.pos += deltaTime * self.spd * self.dir
        # self.pos[0] = max(0., min(world.width, self.pos[0]))
        # self.pos[1] = max(0., min(world.height, self.pos[1]))
        # self.randomRotate(45, 1., deltaTime)
        pass

    def randomRotate(self, range, interval, deltaTime):
        self.interval += deltaTime

        if self.interval >= interval:
            self.interval = 0
            self.degree = random.randrange(-range, range)

        self.rotate(self.degree * deltaTime)
    
    def onCollision(self, food):
        #print('food pos :', food.pos)
        food.isActive = False


class World:
    def __init__(self) -> None:
        self.width = 400
        self.height = 400

        self.objects = []
        self.agents = []
        self.foods = []

    def update(self, window, deltaTime):
        for obj in self.objects:
            if obj.isActive == False:
                continue
            obj.update(self, deltaTime)
            window.blit(obj.sprite, obj.pos - obj.radius)

    def setupObjects(self):
        center = np.array([self.width/2, self.height/2])
        for agent in self.agents:
            agent.lookAt(center)
            agent.rotate(random.randrange(-60, 60))
            # 조건을 만족하면 복사하는 코드를 추가해야한다.

        for food in self.foods:
            food.isActive = True
            food.pos = np.array(
                [random.randrange(0, self.width), random.randrange(0, self.height)])

    def setup(self, agentCount, foodCount):
        sideAgentNum = int(agentCount / 4)
        lastNum = sideAgentNum + (agentCount % 4)
        agentSprite = pygame.image.load('agent.png')
        agent = Agent(agentSprite, [0, 0])
        self.objects.append(agent)
        self.agents.append(agent)

        food_x_List = []
        foodsprite = pygame.image.load('food.png')
        for i in range(foodCount):
            x = random.randrange(0, self.width)
            y = random.randrange(0, self.height)
            food = Food(foodsprite, pos = [x, y])
            self.objects.append(food)
            # self.foods.append(food)
            food_x_List.append([x, food])

        # foods를 x에 대하여 정렬
        food_x_List.sort(key=lambda x: x[0])
        self.foods_x = [x[0] for x in food_x_List]
        self.foods = [i[1] for i in food_x_List]

        #self.setupObjects()

    def checkCollision(self, agent, food):
        return np.linalg.norm(agent.pos - food.pos) < agent.radius + food.radius

    def updateCollision(self):
        for agent in self.agents:
            i = bisect_left(self.foods_x, agent.pos[0])

            if i < len(self.foods_x) and self.checkCollision(agent, self.foods[i]):
                agent.onCollision(self.foods[i])
                del self.foods_x[i], self.foods[i]
            elif (i > 0 and self.checkCollision(agent, self.foods[i-1])):
                agent.onCollision(self.foods[i-1])
                del self.foods_x[i-1], self.foods[i-1]
            

pygame.init()

window = pygame.display.set_mode((500, 500))
backGround = pygame.image.load('backGround.png')

world = World()
world.setup(1, 10)
clock = pygame.time.Clock()
player = world.agents[0]

isRunning = True
while isRunning:
    deltaTime = clock.tick(30) / 100.
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            isRunning = False
        
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                player.pos[0] -= 50 * deltaTime
            if event.key == pygame.K_RIGHT:
                player.pos[0] += 50 * deltaTime
            if event.key == pygame.K_UP:
                player.pos[1] -= 50 * deltaTime
            if event.key == pygame.K_DOWN:
                player.pos[1] += 50 * deltaTime
        
            print(player.pos)
                    
    window.blit(backGround, [0, 0])
    world.updateCollision()
    world.update(window, deltaTime)
    pygame.display.update()

pygame.quit()
