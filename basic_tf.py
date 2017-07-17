#!/usr/bin/env python

import tensorflow as tf
from model import Model
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.rnn import MultiRNNCell
from singen import SinGen
import argparse

lstm_units = 64  # lstm units decides how many outputs
lstm_timesteps = 100  # lstm timesteps is how big to train on
lstm_batchsize = 128


class TSModel(Model):
    '''Basic timeseries tensorflow lstm model'''

    def __init__(self, name, timesteps):
        super().__init__(name)
        self.timesteps = timesteps
        self._build(self.build)

    def build(self):
        # If no input then build a placeholder expecing feed_dict
        with tf.variable_scope('input'):
            self.add(tf.placeholder(tf.float32, (None, self.timesteps, 1)))

        with tf.variable_scope('lstm'):

            multi_cell = MultiRNNCell([LSTMCell(lstm_units), LSTMCell(1)])
            outputs, state = tf.nn.dynamic_rnn(cell=multi_cell, inputs=self.output,
                                               dtype=tf.float32)
            self.add(outputs)
            self.state = state

        with tf.variable_scope('training'):
            # Now make the training bits
            labels = tf.placeholder(tf.float32, [None, self.timesteps, 1], name='labels')
            loss = tf.losses.mean_squared_error(labels, self.output)
            return (labels, self.output, tf.train.AdamOptimizer, loss)


def train(m, epochs, lr):
    g = SinGen(timesteps=lstm_timesteps, batchsize=lstm_batchsize)

    m.set_lr(lr)

    for i in range(epochs):
        print('------------------------------------------')
        print(i)
        print('------------------------------------------')
        x, y = g.batch()
        m.fit(x, y, epochs=10)


def main(_):
    p = argparse.ArgumentParser()
    p.add_argument("--save", help="Name to save snapshot as")
    args = p.parse_args()

    name = args.save
    if name is None:
        name = "NONAME"
    m = TSModel(name=name, timesteps=lstm_timesteps)

    train(m, 384, 1e-3)
    train(m, 12, 1e-4)

    if args.save:
        m.save(args.save)

if __name__ == '__main__':
    tf.app.run()
