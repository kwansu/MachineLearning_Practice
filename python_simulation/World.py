from math import *
from bisect import bisect_left
from typing import Generator
import pygame
import random
import numpy as np

from Object import Agent, Food


class World:
    def __init__(self, width, height) -> None:
        self.worldTime = 0.0
        self.width = width
        self.height = height
        self.generationInterval = 20.

        self.objects = []
        self.agents = []
        self.deadAgents = []  # 오브젝트 풀링용
        self.foods = []
        # 충돌 감지용
        self.foods_col = []
        self.foods_x = []

        self.backGround = pygame.image.load('backGround.png')
        self.agnetSprite = pygame.image.load('agent.png')
        self.foodSprite = pygame.image.load('food.png')

    def update(self, window, deltaTime):
        self.worldTime += deltaTime

        self.updateCollision()

        window.blit(self.backGround, [0, 0])

        for obj in self.foods_col:
            obj.update(self, deltaTime)
            window.blit(self.foodSprite, obj.pos - obj.radius)

        for obj in self.agents:
            if obj.isActive == False:
                continue
            obj.update(self, deltaTime)
            window.blit(obj.sprite, obj.pos - obj.radius)

        # 한 세대의 시간만큼 지나면 전부 초기화후 유전을 발생킨다
        if self.worldTime > self.generationInterval:
            self.worldTime = 0
            self.setupObjects()

    def inheritGenerations(self, generator, deadCount):
        agent = None
        # 오브젝트 풀링
        if len(self.deadAgents) > 0:
            agent = self.deadAgents.pop()
        else:
            agent = Agent(self.agnetSprite)
            self.objects.append(agent)
            self.agents.append(agent)

        agent.isActive = True
        agent.eatFoodCount = deadCount+1
        if generator.pos[0] == 0:
            agent.pos = np.array([0, self.distance(generator.pos[1], 1)])
        else:
            agent.pos = np.array([self.distance(generator.pos[1]), 0])

        # 여기서 유전 형질 변환을 시킨다.
        speed = generator.speed + random.randrange(-10, 10)
        maxDegree = max(1, generator.maxDegree + random.randrange(-5, 5))
        tt = generator.turnaroundTime + random.randrange(-10, 10) * 0.1
        agent.init(self.agnetSprite, speed=speed,
                   maxDgree=maxDegree, turnaroundTime=tt)

    def distance(self, l, axis=0):
        halfAndHalf = self.width/4 if axis == 0 else self.height/4
        return l/2 + halfAndHalf

    def setupObjects(self, deadCount=0, eatFoodCount=2):
        center = np.array([self.width/2, self.height/2])
        for agent in self.agents:
            if agent.isActive == False:
                continue

            # 음식을 못 먹으면 죽인다
            if agent.eatFoodCount <= deadCount:
                agent.isActive = False
                self.deadAgents.append(agent)
                continue

            # 조건을 만족하면 유전시킨다.
            if agent.eatFoodCount >= eatFoodCount:  # 음식을 2개이상 먹으면
                self.inheritGenerations(generator=agent, deadCount=deadCount)

        generationInterval = 5.  # 최소 재생시간

        for agent in self.agents:
            if agent.isActive == False:
                continue

            agent.setup()
            generationInterval = max(
                generationInterval, agent.energe/agent.cost)

            # 일단 가장 가까운 벽에 붙인다
            rightD = self.width - agent.pos[0]
            downD = self.height - agent.pos[1]
            mc = [0, agent.pos[0]] if agent.pos[0] < rightD else [1, rightD]
            mc = mc if agent.pos[1] > mc[1] else [2, agent.pos[1]]
            mc = mc if downD > mc[1] else [3, downD]

            if mc[0] == 0:
                agent.pos[0] = 0
            elif mc[0] == 1:
                agent.pos[0] = self.width
            elif mc[0] == 2:
                agent.pos[1] = 0
            else:
                agent.pos[1] = self.height

            # 중앙을 보게한 후 살짝 각도를 튼다
            agent.lookAt(center)
            agent.rotate(random.randrange(-60, 60))

        self.generationInterval = generationInterval
        print('generationTime :', generationInterval)

        food_x_List = []
        for food in self.foods:
            x = random.randrange(0, self.width)
            y = random.randrange(0, self.height)
            food.isActive = True
            food.pos = np.array([x, y])
            food_x_List.append([x, food])

        # foods를 x에 대하여 정렬
        self.foods_col.clear()
        self.foods_x.clear()
        food_x_List.sort(key=lambda x: x[0])
        self.foods_x = [x[0] for x in food_x_List]
        self.foods_col = [i[1] for i in food_x_List]

        print('agent :', len(self.agents) - len(self.deadAgents))

    def setup(self, agentCount, foodCount):
        for i in range(foodCount):
            food = Food()
            self.objects.append(food)
            self.foods.append(food)

        sideAgentNum = int(agentCount / 4)
        lastNum = sideAgentNum + (agentCount % 4)
        for i in range(agentCount):
            if i < sideAgentNum:
                x = self.width / (sideAgentNum + 1) * (i + 1)
                y = 0.
            elif i < sideAgentNum*2:
                x = self.width
                y = self.height / (sideAgentNum + 1) * (i+1 - sideAgentNum)
            elif i < sideAgentNum*3:
                x = self.width / (sideAgentNum + 1) * (i+1 - sideAgentNum*2)
                y = self.height
            else:
                x = 0.
                y = self.height / (lastNum + 1) * (i+1 - sideAgentNum*3)

            agent = Agent(self.agnetSprite, pos=[x, y])
            self.objects.append(agent)
            self.agents.append(agent)

        self.setupObjects(deadCount=-1)

    def checkCollision(self, agent, food):
        return np.linalg.norm(agent.pos - food.pos) < agent.radius + food.radius

    def updateCollision(self):
        for agent in self.agents:
            if agent.isActive == False:
                continue

            i = bisect_left(self.foods_x, agent.pos[0])

            if i < len(self.foods_x) and self.checkCollision(agent, self.foods_col[i]):
                agent.onCollision(self.foods_col[i])
                del self.foods_x[i], self.foods_col[i]
            elif (i > 0 and self.checkCollision(agent, self.foods_col[i-1])):
                agent.onCollision(self.foods_col[i-1])
                del self.foods_x[i-1], self.foods_col[i-1]
