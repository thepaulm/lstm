#!/usr/bin/env python

import numpy as np
import tensorflow as tf
from model import Model
from tensorflow.contrib.rnn import BasicLSTMCell
# from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.rnn import MultiRNNCell
# from tensorflow.layers import dense
from singen import SinGen
from tensorflow.contrib.layers import fully_connected
import functools

# lstm_units = 3  # lstm units decides how many outputs
# lstm_cells = 4   # lstm cells is how many cells in the state
lstm_timesteps = 22  # lstm timesteps is how big to train on
lstm_batchsize = 128

tf.logging.set_verbosity(tf.logging.INFO)


def get_incoming_shape(incoming):
    """ Returns the incoming data shape """
    if isinstance(incoming, tf.Tensor):
        return incoming.get_shape().as_list()
    elif type(incoming) in [np.array, list, tuple]:
        return np.shape(incoming)
    else:
        raise Exception("Invalid incoming layer.")


def time_distributed(incoming, fn, args=None, scope=None):
    """
    XXX: PAM - this is edit from the original version from tflearn. It has problems
         - it does not work with python3.

    ------------------------- original -------------------------------------
    Time Distributed.

    This layer applies a function to every timestep of the input tensor. The
    custom function first argument must be the input tensor at every timestep.
    Additional parameters for the custom function may be specified in 'args'
    argument (as a list).

    Examples:
        ```python
        # Applying a fully_connected layer at every timestep
        x = time_distributed(input_tensor, fully_connected, [64])

        # Using a conv layer at every timestep with a scope
        x = time_distributed(input_tensor, conv_2d, [64, 3], scope='tconv')
        ```

    Input:
        (3+)-D Tensor [samples, timestep, input_dim].

    Output:
        (3+)-D Tensor [samples, timestep, output_dim].

    Arguments:
        incoming: `Tensor`. The incoming tensor.
        fn: `function`. A function to apply at every timestep. This function
            first parameter must be the input tensor per timestep. Additional
            parameters may be specified in 'args' argument.
        args: `list`. A list of parameters to use with the provided function.
        scope: `str`. A scope to give to each timestep tensor. Useful when
            sharing weights. Each timestep tensor scope will be generated
            as 'scope'-'i' where i represents the timestep id. Note that your
            custom function will be required to have a 'scope' parameter.

    Returns:
        A Tensor.

    """
    if not args:
        args = list()
    assert isinstance(args, list), "'args' must be a list."

    if not isinstance(incoming, tf.Tensor):
        incoming = tf.transpose(tf.stack(incoming), [1, 0, 2])

    input_shape = get_incoming_shape(incoming)
    timestep = input_shape[1]
    x = tf.unstack(incoming, axis=1)
    if scope:
        x = [fn(x[i], scope=scope+'-'+str(i), *args)
             for i in range(timestep)]
    else:
        x = [fn(x[i], *args) for i in range(timestep)]
    try:
        x = [tf.reshape(t, [-1, 1]+get_incoming_shape(t)[1:]) for t in x]
    except:
        x = list(map(lambda t: tf.reshape(t, [-1, 1]+get_incoming_shape(t)[1:]), x))
    return tf.concat(values=x, axis=1)


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


class TSModel(Model):
    '''Basic timeseries tensorflow lstm model'''

    def __init__(self, timesteps, feed_state=True):
        super().__init__()
        self.timesteps = timesteps
        self.feed_state = feed_state
        self._build(self.build)

    def build(self):
        # If no input then build a placeholder expecing feed_dict
        with tf.variable_scope('input'):
            self.add(tf.placeholder(tf.float32, (None, self.timesteps, 1)))

        with tf.variable_scope('lstm'):
            # XXX - fix state: https://www.tensorflow.org/tutorials/recurrent

            initial_state = None
            if self.feed_state:
                # XXX feed this back eventually - this will be like stateful in keras
                self.input_state = tf.placeholder(tf.float32, [self.timesteps, 2, lstm_batchsize, 1])
                cht = tf.unstack(self.input_state, axis=0)
                rnn_tuple_state = tuple([tf.contrib.rnn.LSTMStateTuple(cht[i][0], cht[i][1]) for i in range(self.timesteps)])
                initial_state = rnn_tuple_state

            # cells = [BasicLSTMCell(num_units=timesteps, state_is_tuple=True) for _ in range(timesteps)]
            # multi_cell = MultiRNNCell(cells=cells, state_is_tuple=True)
            cells = [BasicLSTMCell(64, forget_bias=0.0) for _ in range(self.timesteps)]
            multi_cell = MultiRNNCell(cells)
            # if initial_state is None:
            #     initial_state = multi_cell.zero_state(1, tf.float32)

            # outputs, state = tf.nn.dynamic_rnn(cell=multi_cell, inputs=self.output,
            #                                    dtype=tf.float32, initial_state=initial_state)

            # XXX manually unroll this
            outputs, state = tf.nn.dynamic_rnn(cell=multi_cell, inputs=self.output,
                                               dtype=tf.float32, initial_state=initial_state)
            variable_summaries(outputs)
            self.add(outputs)
            self.state = state

        with tf.variable_scope('time_distributed'):
            # Add time distributed fully connected layers
            fcrelu = functools.partial(fully_connected, activation_fn=tf.nn.relu)
            self.add(time_distributed(self.output, fcrelu, [1]))

        with tf.variable_scope('training'):
            # Now make the training bits
            self.labels = tf.placeholder(tf.float32, [lstm_batchsize, self.timesteps, 1], name='labels')
            self.loss = tf.losses.mean_squared_error(self.labels, self.output)

            self.global_step = tf.contrib.framework.get_or_create_global_step()
            self.train_op = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(self.loss,
                                                                                global_step=self.global_step)

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
                    (_, state) = sess.run([m.train_op, m.state], feed_dict={m.input: x, m.labels: y, m.input_state: state})
                print("10 epochs")

    return rvh.get_losses()


def nostate_train(m, epochs, log_every=1, log_predictions=False):

    rvh = get_rvh(m, log_every, log_predictions)
    # hooks = [tf.train.StopAtStepHook(last_step=epochs), rvh, ]
    hooks = [rvh]

    with m.graph.as_default():
        tf.summary.merge_all()
        with tf.summary.FileWriter('./tf_tblogs', m.graph):
            with tf.train.SingularMonitoredSession(hooks=hooks, config=get_sess_config()) as sess:
                g = SinGen(timesteps=m.timesteps, batchsize=lstm_batchsize)
                while not sess.should_stop():
                    x, y = g.batch()
                    for _ in range(10):
                        sess.run([m.train_op], feed_dict={m.input: x, m.labels: y})
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


def train_two():
    m2 = TSModel(timesteps=lstm_timesteps, feed_state=False)
    print(m2)
    print(nostate_train(m2, 48, log_every=10, log_predictions=False))


def main(_):
    # compare_models()
    # train_two()
    train_two()

if __name__ == '__main__':
    tf.app.run()
