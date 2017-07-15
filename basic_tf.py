#!/usr/bin/env python

import numpy as np
import tensorflow as tf
from model import Model
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.rnn import MultiRNNCell
from singen import SinGen
import signal
import sys
import argparse

# lstm_cells = 4   # lstm cells is how many cells in the state
lstm_units = 64  # lstm units decides how many outputs
lstm_timesteps = 22  # lstm timesteps is how big to train on
lstm_batchsize = 128
default_lr = 1e-3

tf.logging.set_verbosity(tf.logging.INFO)

should_exit = False


class TSModel(Model):
    '''Basic timeseries tensorflow lstm model'''

    def __init__(self, name, timesteps, batchsize, lr=default_lr, feed_state=True):
        super().__init__(name)
        self.timesteps = timesteps
        self.feed_state = feed_state
        self.lr = lr
        self.batchsize = batchsize
        self._build(self.build)

    def build(self):
        print("Building with learning rate %f\n" % self.lr)
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
            self.labels = tf.placeholder(
                tf.float32, [None, self.timesteps, 1], name='labels')
            self.loss = tf.losses.mean_squared_error(self.labels, self.output)
            tf.summary.scalar('label', self.labels[0][0][0])
            tf.summary.scalar('predict', self.output[0][0][0])

            self.global_step = tf.contrib.framework.get_or_create_global_step()
            self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss,
                                                                                   global_step=self.global_step)

        self.summary = tf.summary.merge_all()

    def __repr__(self):
        out = super().__repr__()
        out += '\n\nstate:\n'
        out += '\n'.join([str(s) for s in self.state])
        return out

        # with tf.variable_scope(sn('linear_out')):
        #     self.add(fully_connected(inputs=self.output, num_outputs=1))


class ReturnValuesHook(tf.train.LoggingTensorHook):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.losses = []

    def after_run(self, run_context, run_values):
        super().after_run(run_context, run_values)
        if run_values is not None and run_values.results is not None:
            self.losses.append(run_values.results['loss'])

    def get_losses(self):
        return self.losses


def get_sess_config():
    config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # config.log_device_placement = True
    return config


def get_rvh(m, log_every, log_predictions):
    log_map = {'step': m.global_step, 'loss': m.loss}
    if log_predictions:
        log_map.update({'label': m.labels, 'output': m.output})
    rvh = ReturnValuesHook(tensors=log_map, every_n_iter=log_every)
    return rvh


def state_train(m, epochs, log_every=1, log_predictions=False):

    rvh = get_rvh(m, log_every, log_predictions)
    # hooks = [tf.train.StopAtStepHook(last_step=epochs), rvh]
    hooks = [rvh]

    with m.graph.as_default():
        state = np.zeros((m.timesteps, 2, lstm_batchsize, 1))
        with tf.train.SingularMonitoredSession(hooks=hooks, config=get_sess_config()) as sess:
            g = SinGen(timesteps=m.timesteps, batchsize=lstm_batchsize)
            while not sess.should_stop():
                x, y = g.batch()
                for _ in range(10):
                    (_, state) = sess.run([m.train_op, m.state], feed_dict={
                        m.input: x, m.labels: y, m.input_state: state})
                print("10 epochs")

    return rvh.get_losses()


def nostate_train(args, m, epochs, log_every=1, log_predictions=False):

    global should_exit

    rvh = get_rvh(m, log_every, log_predictions)
    # hooks = [tf.train.StopAtStepHook(last_step=epochs), rvh, ]
    hooks = [rvh]

    with m.graph.as_default():
        saver = None
        if args.name is not None:
            saver = tf.train.Saver()

        # fw = tf.summary.FileWriter('./tf_tblogs', m.graph)
        with tf.train.SingularMonitoredSession(hooks=hooks, config=get_sess_config()) as sess:

            if args.name is not None:
                try:
                    saver.restore(sess.raw_session(),
                                  "checkpoints/" + args.name)
                except:
                    pass

            g = SinGen(timesteps=m.timesteps, batchsize=lstm_batchsize)
            while not sess.should_stop():
                if should_exit:
                    if saver is not None:
                        print("Give me a few, I'm gonna save this model ...")
                        saver.save(sess.raw_session(),
                                   "checkpoints/" + args.name)
                        print("done.")
                    sys.exit(0)
                x, y = g.batch()
                for i in range(10):
                    (s, o) = sess.run([m.summary, m.train_op],
                                      feed_dict={m.input: x, m.labels: y})
                    # fw.add_summary(s, i)
                print("10 epochs")

    return rvh.get_losses()


def compare_models():
    m = TSModel(timesteps=lstm_timesteps)
    print(state_train(m, 5))

    m2 = TSModel(timesteps=lstm_timesteps, feed_state=False)
    print(nostate_train(m2, 5))


def train_one():
    m = TSModel(timesteps=lstm_timesteps)
    print(m)
    print(nostate_train(m, 48, log_every=1))


def train_two(args):
    m2 = TSModel(name=args.name, batchsize=lstm_batchsize,
                 timesteps=lstm_timesteps, lr=args.lr, feed_state=False)
    print(m2)
    print(nostate_train(args, m2, 48, log_every=10, log_predictions=False))


def handle_ctrl_c():
    def f(sig, _):
        global should_exit
        should_exit = True
    signal.signal(signal.SIGINT, f)


def main(_):
    handle_ctrl_c()

    p = argparse.ArgumentParser()
    p.add_argument(
        "--name", help="Name for this training (will load and re-save weights)")
    p.add_argument("--lr", type=float, default=default_lr,
                   help="Learning rate")
    args = p.parse_args()
    # compare_models()
    # train_two()
    train_two(args)


if __name__ == '__main__':
    tf.app.run()
