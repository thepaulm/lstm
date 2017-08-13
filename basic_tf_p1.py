#!/usr/bin/env python

import tensorflow as tf
from model import Model
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.rnn import MultiRNNCell
from singen import SinP1Gen
import argparse

lstm_units = 64  # lstm units decides how many outputs
lstm_timesteps = 100  # lstm timesteps is how big to train on
lstm_batchsize = 1024

default_lr = 5e-2
default_iterations = 64

# best:
# ./basic_tf.py --iterations 128 --lr 5e-3 --tensorboard-dir tflstm --save tflstm/iter128_5e-3


def summary_name(s):
    return s.replace(':', '_')


class TSModel(Model):
    '''Basic timeseries tensorflow lstm model'''

    def __init__(self, name, timesteps, l2norm=True, breadth=1, depth=2, linear=1,
                 tensorboard_dir=None):
        super().__init__(name, tensorboard_dir=tensorboard_dir)
        self.timesteps = timesteps
        self.breadth = breadth
        self.l2norm = l2norm
        self.depth = depth
        self.linear = linear
        self.build(self.mybuild)

    def mybuild(self):
        # If no input then build a placeholder expecing feed_dict
        with tf.variable_scope('input'):
            self.add(tf.placeholder(tf.float32, (None, self.timesteps, 1)))

            if self.l2norm:
                self.add(tf.nn.l2_normalize(self.output, dim=1))

        # every one but the last one is 64 units
        for i in range(self.depth - 1):
            with tf.variable_scope('lstm%d' % i):
                multi_cell = MultiRNNCell([LSTMCell(lstm_units) for _ in range(self.breadth)])
                outputs, z = tf.nn.dynamic_rnn(cell=multi_cell, inputs=self.output,
                                               dtype=tf.float32)
                self.add(outputs)

        with tf.variable_scope('lstm%d' % (self.depth - 1)):
            cells = [LSTMCell(lstm_units) for _ in range(self.breadth - 1)] + [LSTMCell(1)]
            multi_cell = MultiRNNCell(cells)
            outputs, z = tf.nn.dynamic_rnn(cell=multi_cell, inputs=self.output,
                                           dtype=tf.float32)
            self.add(outputs)

        for i in range(self.linear):
            with tf.variable_scope('linear%d' % i):
                l = tf.layers.dense(self.output, units=1, activation=None)
                self.add(l)

        with tf.variable_scope('training'):
            # Now make the training bits
            labels = tf.placeholder(
                tf.float32, [None, self.timesteps, 1], name='labels')
            loss = tf.losses.mean_squared_error(labels, self.output)
            return (labels, self.output, tf.train.AdamOptimizer, loss)


def train(m, epochs, lr, epere=10, verbose=True):
    g = SinP1Gen(timesteps=lstm_timesteps, batchsize=lstm_batchsize)

    m.set_lr(lr)

    losses = []
    for i in range(epochs):
        if verbose:
            print('------------------------------------------')
            print(i)
            print('------------------------------------------')
        x, y = g.batch()
        l = m.fit(x, y, epochs=epere, verbose=verbose)
        losses.extend(l)

    return losses


def main(_):
    p = argparse.ArgumentParser()
    p.add_argument("--save", help="Name to save snapshot as")
    p.add_argument("--tensorboard-dir",
                   help="Directory to write tensorboard snapshots")
    p.add_argument("--iterations", help="iterations to run", type=int,
                   default=default_iterations)
    p.add_argument("--lr", help="learning rate",
                   type=float, default=default_lr)
    p.add_argument("--l2norm", help="l2 normalization on input",
                   default=False, action="store_true")
    p.add_argument("--breadth", help="Lstm cells per layer", type=int)
    p.add_argument("--depth", help="Lstm cell layers", type=int)
    p.add_argument("--linear", help="Linear layers", type=int)
    p.add_argument("--epere", help="epochs per epoch", type=int, default=10)
    args = p.parse_args()

    name = args.save
    if name is None:
        name = "NONAME"
    m = TSModel(name=name, timesteps=lstm_timesteps, l2norm=args.l2norm, breadth=args.breadth,
                depth=args.depth, linear=args.linear, tensorboard_dir=args.tensorboard_dir)
    print(m)
    print("Training %d iterations with lr %f" % (args.iterations, args.lr))
    train(m, args.iterations, args.lr, epere=args.epere)

    if args.save:
        m.save(args.save)


if __name__ == '__main__':
    tf.app.run()
