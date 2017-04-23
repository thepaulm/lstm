import math
import numpy as np


class SinGen(object):
    def __init__(self, start=0.0, step=1.0, timesteps=10):
        self.start = start
        self.step = step
        self.timesteps = timesteps
        self.x = start

    def batch(self):
        '''
        batch

        Produce input and labels array shape (timesteps, 1)
        (for feeding into input tensor shape (None, timesteps, 1))
        '''
        proj = []
        for _ in range(self.timesteps + 1):
            proj.append(math.sin(self.x))
            self.x += self.step
        xs = proj[:-1]
        ys = proj[1:]
        outs = (1, self.timesteps, 1)
        return np.array(xs).reshape(outs), np.array(ys).reshape(outs)
