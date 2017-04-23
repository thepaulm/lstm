#!/usr/bin/env python

from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers import TimeDistributed, Dense
from keras.optimizers import Adam
from singen import SinGen


lstm_timesteps = 22  # lstm timesteps is how big to train on
lstm_batchsize = 128
lstm_units = 64


class TSModel(object):
    def __init__(self, timesteps):
        self.m = Sequential()
        # You can add two of these later
        self.m.add(LSTM(lstm_units, return_sequences=True,
                        input_shape=(timesteps, 1)))
        self.m.add(TimeDistributed(Dense(1)))
        self.m.compile(loss='mean_squared_error', optimizer=Adam())


def train(m, epochs, lr):
    m.m.optimizer.lr = lr
    g = SinGen(timesteps=lstm_timesteps, batchsize=lstm_batchsize)
    for i in range(epochs):
        print('------------------------------------------')
        print(i)
        print('------------------------------------------')
        x, y = g.batch()
        m.m.fit(x, y, batch_size=lstm_batchsize, epochs=10)


def main():
    m = TSModel(timesteps=lstm_timesteps)
    train(m, 48, 1e-3)
    train(m, 22, 1e-4)
    train(m, 22, 1e-5)

    # m.m.save_weights('keras_lstm.h5')

if __name__ == '__main__':
    main()
