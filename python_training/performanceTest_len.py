import time

def checkPerformance(func, loopCount):
    start_time = time.time()
    start = time.gmtime(start_time)
    print("측정 시작")

    func(loopCount)

    end_time = time.time()
    end = time.gmtime(end_time)
    print("측정 완료")
    
    # 소요 시간 측정
    end_start = end_time - start_time
    end_start = time.gmtime(end_start)
    print("소요시간 : %d시 %d분 %d초"%(end_start.tm_hour, end_start.tm_min, end_start.tm_sec))

l = tuple(range(10))

def processA(loopCount):
    for i in range(loopCount):
        a = l[5:len(l)]
        

def processB(loopCount):
    for i in range(loopCount):
        a = l[5:-1]

count = 100000000
checkPerformance(processA, count)
checkPerformance(processB, count)
checkPerformance(processA, count)
checkPerformance(processB, count)
