import numpy
import tensorflow

data = numpy.array([[828.659973, 833.450012, 908100, 828.349976, 831.659973],
                    [823.02002, 828.070007, 1828100, 821.655029, 828.070007],
                    [819.929993, 824.400024, 1438100, 818.97998, 824.159973],
                    [816, 820.958984, 1008100, 815.48999, 819.23999],
                    [819.359985, 823, 1188100, 818.469971, 818.97998],
                    [819, 823, 1198100, 816, 820.450012],
                    [811.700012, 815.25, 1098100, 809.780029, 813.669983],
                    [809.51001, 816.659973, 1398100, 804.539978, 809.559998]])

# 아래의 정규화 과정이 없이 하면 크기차이로 학습이 재대로 되지않는다.
'''
numerator = data - numpy.min(data, 0)
denominator = numpy.max(data, 0) - numpy.min(data, 0)
data = numerator / (denominator + 1e-7)
'''
# standardization
data = (data - data.mean())/data.std()


x_data = data[:, 0:-1]
y_data = data[:, [-1]]

model = tensorflow.keras.models.Sequential(
    [tensorflow.keras.layers.Dense(1, activation='linear')])
model.compile(optimizer=tensorflow.keras.optimizers.SGD(
    learning_rate=0.01), loss='mse')

result = model.fit(x_data, y_data, epochs=1000)
score = model.evaluate(x_data, y_data)
print('Cost: ', score)
