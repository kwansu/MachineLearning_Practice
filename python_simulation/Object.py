from math import *
from pygame import *
import random
import numpy as np
import pygame


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
    def __init__(self, pos=[0., 0.], radius=10.) -> None:
        super().__init__(pos, radius)


class Agent(Object):
    def __init__(self, sprite, pos=[0., 0.], radius=15.) -> None:
        super().__init__(pos, radius)
        self.dir = np.array([1, 0])  # 항상 normalize 시킬것
        self.init(sprite)
        self.setup()

    def init(self, sprite, speed=100.0, maxDgree=60.0, turnaroundTime=1.0):
        self.speed = speed
        self.cost = 10 * ((speed/100)**2)  # 계산 생략을 위한 연비
        self.turnaroundTime = turnaroundTime
        self.maxDegree = maxDgree
        red = max(0,min(127, int(speed) - 80)) * 2
        self.changColor(sprite, (red, 0, 255 - red, 0))

    def setup(self):
        self.energe = 100.  # 이동시 사용, 에너지 없으면 움직이지 못한다.
        self.eatFoodCount = 0
        self.interval = 0.
        self.degree = 0.

    def rotate(self, degree):
        r = radians(degree)
        mat = np.array([[cos(r), -sin(r)], [sin(r), cos(r)]])
        self.dir = np.dot(self.dir, mat)
        self.dir /= np.linalg.norm(self.dir)

    def lookAt(self, atPos):
        self.dir = atPos - self.pos
        self.dir /= np.linalg.norm(self.dir)

    def update(self, world, deltaTime):
        if self.energe > 0:
            self.energe -= self.cost * deltaTime
            self.pos += deltaTime * self.speed * self.dir
            self.pos[0] = max(0., min(world.width, self.pos[0]))
            self.pos[1] = max(0., min(world.height, self.pos[1]))
            self.randomRotate(self.maxDegree, self.turnaroundTime, deltaTime)

    def randomRotate(self, range, interval, deltaTime):
        self.interval += deltaTime

        if self.interval >= interval:
            self.interval = 0
            self.degree = random.randrange(-range, range)

        self.rotate(self.degree * deltaTime)

    def onCollision(self, food):
        food.isActive = False
        self.eatFoodCount += 1

    def changColor(self, image, color):
        colouredImage = Surface(image.get_size())
        colouredImage.fill(color)

        self.sprite = image.copy()
        self.sprite.blit(colouredImage,(0,0), special_flags=BLEND_MULT)
