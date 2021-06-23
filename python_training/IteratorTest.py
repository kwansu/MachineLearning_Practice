class TimeIterator:
    def __init__(self, start, stop):
        self.start = start
        self.current = start
        self.stop = stop
    def __iter__(self):
        return self
    def __getTime(self, s):
        hour, minute = divmod(s, 3600)
        hour = hour % 24
        minute, second = divmod(minute, 60)
        return "{:02d}:{:02d}:{:02d}".format(hour, minute, second)
    def __next__(self):
        if self.current >= self.stop:
            raise StopIteration
        str = self.__getTime(self.current)
        self.current += 1
        return str
    def __getitem__(self, index):
        if index < self.stop:
            return self.__getTime(self.start+index)
        else:
            raise IndexError

start, stop, index = (88234, 88237, 1)
 
for i in TimeIterator(start, stop):
    print(i)
 
print('\n', TimeIterator(start, stop)[index], sep='')