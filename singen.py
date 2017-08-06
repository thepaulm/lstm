import math
import numpy as np


class FGen(object):
    def __init__(self, fun, start=0.0, step=1.0, timesteps=10, batchsize=1):
        self.start = start
        self.step = step
        self.timesteps = timesteps
        self.x = start
        self.batchsize = batchsize
        self.fun = fun

    def batch(self):
        '''
        batch

        Produce input and labels array shape (timesteps, 1)
        (for feeding into input tensor shape (None, timesteps, 1))
        '''
        xbatches = []
        ybatches = []
        for _ in range(self.batchsize):
            xs = []
            for _ in range(self.timesteps):
                xs.append(self.fun(self.x))
                self.x += self.step
            extra = self.fun(self.x)  # Extra for the y space - don't move x
            ys = xs[1:] + [extra]
            xbatches.append(xs)
            ybatches.append(ys)
        outs = (self.batchsize, self.timesteps, 1)
        return np.array(xbatches).reshape(outs), np.array(ybatches).reshape(outs)


class SinGen(FGen):
    def __init__(self, start=0.0, step=1.0, timesteps=10, batchsize=1):
        super().__init__(lambda x: math.sin(x), start=start, step=step, timesteps=timesteps, batchsize=batchsize)


class SinP1Gen(FGen):
    def __init__(self, start=0.0, step=1.0, timesteps=10, batchsize=1, mul=0.10):
        super().__init__(self, start=start, step=step, timesteps=timesteps, batchsize=batchsize)
        self.mul = mul

    def __call__(self, x):
        return math.sin(x) + x * self.mul
