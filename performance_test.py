import numpy as np
from time import time
from functools import wraps


def print_execution_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time()
        func(*args, **kwargs)
        print(f'{func.__name__}, execution time : {time() - start_time}')
    return wrapper


@print_execution_time
def test_clip(tensor : np.ndarray, loop_count):
    temp = None
    for _ in range(loop_count):
        temp = tensor.clip(min = 0.)


@print_execution_time
def test_where(tensor : np.ndarray, loop_count):
    temp = None
    for _ in range(loop_count):
        temp = np.where(tensor > 0., tensor, 0.)


# a = np.random.random((100,100,100,100)) - 0.5
# loop_count = 10

# test_clip(a, loop_count)
# test_where(a, loop_count)
# test_clip(a, loop_count)
# test_where(a, loop_count)
