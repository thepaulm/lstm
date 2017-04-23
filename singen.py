import math
import numpy as np


class SinGen(object):
    def __init__(self, start=0.0, step=1.0, timesteps=10, batchsize=1):
        self.start = start
        self.step = step
        self.timesteps = timesteps
        self.x = start
        self.batchsize = batchsize

    def batch(self):
        '''
        batch

        Produce input and labels array shape (timesteps, 1)
        (for feeding into input tensor shape (None, timesteps, 1))
        '''
        xbatches = []
        ybatches = []
        for _ in range(self.batchsize):
            proj = []
            for _ in range(self.timesteps + 1):
                proj.append(math.sin(self.x))
                self.x += self.step
            xs = proj[:-1]
            ys = proj[1:]
            xbatches.append(xs)
            ybatches.append(ys)
        outs = (self.batchsize, self.timesteps, 1)
        return np.array(xbatches).reshape(outs), np.array(ybatches).reshape(outs)
