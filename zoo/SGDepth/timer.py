import time

class Timer(object):
    def __init__(self):
        self.time_prev = time.monotonic()
        self.category_prev = 'unaccounted'
        self.accumulators = dict()

    def enter(self, name):
        now = time.monotonic()
        delta = now - self.time_prev
        self.time_prev = now

        if self.category_prev not in self.accumulators:
            self.accumulators[self.category_prev] = 0

        self.accumulators[self.category_prev] += delta

        self.category_prev = name

    def leave(self):
        self.enter('unaccounted')

    def items(self):
        return self.accumulators.items()
