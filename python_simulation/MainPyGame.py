import pygame

class MainPygame:
    def __init__(self, width, height, speed = 4, fps = 0, mimDeltaTime = 0.5) -> None:
        self.width = width
        self.height = height

        pygame.init()
        self.window = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()
        # 재생속도를 너무 높일 경우 이동,회전,충돌 물리계산에 문제가 생길 수 있다.
        self.timeSpeed = 0.001 * speed # 1배속 = 0.001
        self.minDeltaTime = mimDeltaTime * 1000
        self.fps = fps
        self.isRunning = True
        self.isStoped = False


    def stop(self, stop = True):
        self.isStoped = True
    

    def run(self, world):
        while self.isRunning:
            deltaTime = self.clock.tick(self.fps)

            if (deltaTime > self.minDeltaTime):
                deltaTime = self.minDeltaTime

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.isRunning = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.isStoped = not self.isStoped
                    if event.key == pygame.K_UP:
                        self.timeSpeed *= 2.0 #2배씩 빨라짐
                        print('재생속도 :', self.timeSpeed*1000)
                    if event.key == pygame.K_DOWN:
                        print('재생속도 :', self.timeSpeed*1000)
                        self.timeSpeed *= 0.5 #0.5배

            if self.isStoped:
                continue
            
            world.update(self.window, deltaTime*self.timeSpeed)
            pygame.display.update()

        pygame.quit()     
