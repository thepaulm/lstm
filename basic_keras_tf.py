#!/usr/bin/env python

import argparse
from tensorflow.contrib.keras.api.keras.models import Sequential
from tensorflow.contrib.keras.api.keras.layers import LSTM
from tensorflow.contrib.keras.api.keras.layers import Dense
from tensorflow.contrib.keras.python.keras.layers.wrappers import TimeDistributed
from tensorflow.contrib.keras.api.keras.optimizers import Adam
from tensorflow.contrib.keras.api.keras.callbacks import TensorBoard
from singen import SinGen

_ = Dense
_ = TimeDistributed


lstm_timesteps = 100  # lstm timesteps is how big to train on
lstm_batchsize = 128
lstm_units = 64


class TSModel(object):
    def __init__(self, timesteps, batchsize):
        self.m = Sequential()

        self.m.add(LSTM(units=lstm_units, return_sequences=True, stateful=False,
                        input_shape=(timesteps, 1)))
        self.m.add(LSTM(units=1, return_sequences=True, stateful=False))
        self.m.compile(loss='mean_squared_error', optimizer=Adam())

        print(self.m.summary())


def train(m, epochs, lr, batchsize, tensorboard):
    m.m.optimizer.lr = lr
    g = SinGen(timesteps=lstm_timesteps, batchsize=batchsize)

    callbacks = None
    if tensorboard is not None:
        callbacks = [TensorBoard(log_dir=tensorboard, histogram_freq=1,
                                 write_graph=True, write_images=True)]
    for i in range(epochs):
        print('------------------------------------------')
        print(i)
        print('------------------------------------------')
        x, y = g.batch()
        print("x shape: ", x.shape)
        m.m.fit(x, y, batch_size=lstm_batchsize, epochs=10,
                callbacks=callbacks
                )


def get_args():
    p = argparse.ArgumentParser("Train Keras LSTM Model for sine wave")
    p.add_argument('--save', help="h5 file to save model to when done")
    p.add_argument('--tensorboard', help="tensorboard log dir")
    return p.parse_args()


def main():
    args = get_args()
    m = TSModel(timesteps=lstm_timesteps, batchsize=lstm_batchsize)
    train(m, 384, 1e-3, lstm_batchsize, args.tensorboard)

    if args.save is not None:
        m.m.save_weights(args.save)

if __name__ == '__main__':
    main()
