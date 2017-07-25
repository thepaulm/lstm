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

default_lr = 1e-2
default_iterations = 64

# best:
# ./basic_tf.py --iterations 128 --lr 5e-3 --tensorboard-dir tflstm --save tflstm/iter128_5e-3


def summary_name(s):
    return s.replace(':', '_')


class TSModel(Model):
    '''Basic timeseries tensorflow lstm model'''

    def __init__(self, name, timesteps, tensorboard_dir=None):
        super().__init__(name, tensorboard_dir=tensorboard_dir)
        self.timesteps = timesteps
        self.build(self.mybuild)

    def mybuild(self):
        # If no input then build a placeholder expecing feed_dict
        with tf.variable_scope('input'):
            self.add(tf.placeholder(tf.float32, (None, self.timesteps, 1)))

        with tf.variable_scope('lstm'):
            multi_cell = MultiRNNCell([LSTMCell(lstm_units), LSTMCell(1)])

            def add_summaries():
                for v in multi_cell.variables:
                    tf.summary.histogram(summary_name(v.name), v)
                for w in multi_cell.weights:
                    tf.summary.histogram(summary_name(w.name), w)

            outputs, state = tf.nn.dynamic_rnn(cell=multi_cell, inputs=self.output,
                                               dtype=tf.float32)
            add_summaries()
            self.add(outputs)
            self.state = state

        with tf.variable_scope('training'):
            # Now make the training bits
            labels = tf.placeholder(
                tf.float32, [None, self.timesteps, 1], name='labels')
            loss = tf.losses.mean_squared_error(labels, self.output)
            return (labels, self.output, tf.train.AdamOptimizer, loss)


def train(m, epochs, lr, verbose=True):
    g = SinGen(timesteps=lstm_timesteps, batchsize=lstm_batchsize)

    m.set_lr(lr)

    losses = []
    for i in range(epochs):
        if verbose:
            print('------------------------------------------')
            print(i)
            print('------------------------------------------')
        x, y = g.batch()
        l = m.fit(x, y, epochs=10, verbose=verbose)
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
    args = p.parse_args()

    name = args.save
    if name is None:
        name = "NONAME"
    m = TSModel(name=name, timesteps=lstm_timesteps,
                tensorboard_dir=args.tensorboard_dir)

    print("Training %d iterations with lr %f" % (args.iterations, args.lr))
    train(m, args.iterations, args.lr)

    if args.save:
        m.save(args.save)


if __name__ == '__main__':
    tf.app.run()
