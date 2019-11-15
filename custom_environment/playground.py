# get pyhanabi observation and ensure that specific state is met, may require

class StorageRewardMetrics(object):
    def __init__(self):
        self._efficiency = None

    @property
    def efficiency(self):
        return self._efficiency

    @efficiency.setter
    def efficiency(self, value):
        self._efficiency = value

s = StorageRewardMetrics()

s.efficiency = 42
print(s.efficiency)

d = {'Y': 1, 'R': 2}
print(sum(d.values()))