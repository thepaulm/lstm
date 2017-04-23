#!/usr/bin/env python

import numpy as np
import tensorflow as tf
from model import Model
from tensorflow.contrib.rnn import BasicLSTMCell
# from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.rnn import MultiRNNCell
from tensorflow.contrib.layers import fully_connected  # noqa
from singen import SinGen

# lstm_units = 3  # lstm units decides how many outputs
# lstm_cells = 4   # lstm cells is how many cells in the state
lstm_timesteps = 22  # lstm timesteps is how big to train on
lstm_batchsize = 1

tf.logging.set_verbosity(tf.logging.INFO)


class TSModel(Model):
    '''Basic timeseries tensorflow lstm model'''

    def __init__(self, timesteps, feed_state=True, inputs=None):
        super().__init__()

        self.graph = tf.Graph()
        sn = self.name
        self.timesteps = timesteps

        with self.graph.as_default():
            # If no input then build a placeholder expecing feed_dict
            if inputs is None:
                with tf.variable_scope(sn('input')):
                    self.add(tf.placeholder(tf.float32, (None, timesteps, 1)))
            else:
                self.output = inputs

            with tf.variable_scope(sn('lstml_1')):
                # XXX - fix state: https://www.tensorflow.org/tutorials/recurrent

                initial_state = None
                if feed_state:
                    self.input_state = tf.placeholder(tf.float32, [timesteps, 2, lstm_batchsize, 1])
                    cht = tf.unstack(self.input_state, axis=0)
                    rnn_tuple_state = tuple([tf.contrib.rnn.LSTMStateTuple(cht[i][0], cht[i][1]) for i in range(timesteps)])
                    initial_state = rnn_tuple_state

                # cells = [BasicLSTMCell(num_units=timesteps, state_is_tuple=True) for _ in range(timesteps)]
                # multi_cell = MultiRNNCell(cells=cells, state_is_tuple=True)
                cells = [BasicLSTMCell(1) for _ in range(timesteps)]
                multi_cell = MultiRNNCell(cells)
                # if initial_state is None:
                #     initial_state = multi_cell.zero_state(1, tf.float32)

                outputs, state = tf.nn.dynamic_rnn(cell=multi_cell, inputs=self.output,
                                                   dtype=tf.float32, initial_state=initial_state)
                self.add(outputs)
                self.state = state

                # Now make the training bits
                self.labels = tf.placeholder(tf.float32, [lstm_batchsize, timesteps, 1], name='labels')
                self.loss = tf.losses.mean_squared_error(self.output, self.labels)
                opt = tf.train.AdamOptimizer(learning_rate=1e-4)
                self.global_step = tf.contrib.framework.get_or_create_global_step()
                self.train_opt = opt.minimize(self.loss, global_step=self.global_step)

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


def state_train(m, epochs, log_every=1):

    rvh = ReturnValuesHook(tensors={'step': m.global_step, 'loss': m.loss,
                                    'label': m.labels, 'output': m.output}, every_n_iter=log_every)
    hooks = [tf.train.StopAtStepHook(last_step=epochs), rvh, ]

    with m.graph.as_default():
        state = np.zeros((m.timesteps, 2, lstm_batchsize, 1))
        with tf.train.SingularMonitoredSession(hooks=hooks, config=get_sess_config()) as sess:
            g = SinGen(timesteps=m.timesteps)
            while not sess.should_stop():
                x, y = g.batch()
                (_, state) = sess.run([m.train_opt, m.state], feed_dict={m.input: x, m.labels: y, m.input_state: state})

    return rvh.get_losses()


def nostate_train(m, epochs, log_every=1):

    rvh = ReturnValuesHook(tensors={'step': m.global_step, 'loss': m.loss,
                                    'label': m.labels, 'output': m.output}, every_n_iter=log_every)
    hooks = [tf.train.StopAtStepHook(last_step=epochs), rvh, ]

    with m.graph.as_default():
        with tf.train.SingularMonitoredSession(hooks=hooks, config=get_sess_config()) as sess:
            g = SinGen(timesteps=m.timesteps)
            while not sess.should_stop():
                x, y = g.batch()
                sess.run([m.train_opt], feed_dict={m.input: x, m.labels: y})

    return rvh.get_losses()


def compare_models():
    m = TSModel(timesteps=lstm_timesteps)
    print(state_train(m, 5))

    m2 = TSModel(timesteps=lstm_timesteps, feed_state=False)
    print(nostate_train(m2, 5))


def train_two():
    m2 = TSModel(timesteps=lstm_timesteps, feed_state=False)
    print(nostate_train(m2, 500, log_every=20))


def main(_):
    # compare_models()
    train_two()

if __name__ == '__main__':
    tf.app.run()
