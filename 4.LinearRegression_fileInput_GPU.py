import numpy
import time
import tensorflow
from tensorflow.python.client import device_lib
device_lib.list_local_devices()

with tensorflow.device("/gpu:0"):
    loadedData = numpy.loadtxt('data/data-01-test-score.csv', delimiter=',', dtype=numpy.float32)
    x_data = loadedData[:, 0:-1]
    y_data = loadedData[:, [-1]]

    model = tensorflow.keras.models.Sequential([tensorflow.keras.layers.Dense(1, activation='linear')])

    model.compile(optimizer = tensorflow.keras.optimizers.SGD(lr=1e-5), loss='mse')

    start_time = time.time()
    start = time.gmtime(start_time)
    print("훈련 시작 : %d시 %d분 %d초"%(start.tm_hour, start.tm_min, start.tm_sec))

    model.fit(x_data, y_data, epochs=1000,verbose=0)

    predict = model.predict(numpy.array([[80, 80, 80]]))
    end_time = time.time()
    end = time.gmtime(end_time)
    print("훈련 끝 : %d시 %d분 %d초"%(end.tm_hour, end.tm_min, end.tm_sec))

    # 소요 시간 측정
    end_start = end_time - start_time
    end_start = time.gmtime(end_start)
    print("소요시간 : %d시 %d분 %d초"%(end_start.tm_hour, end_start.tm_min, end_start.tm_sec))
