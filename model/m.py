#!/usr/bin/env python

import tensorflow as tf


class Model(object):
    '''
    Base Model for building tensorflow graphs.

    Example usage: TBD
    '''
    def __init__(self, name):
        self.layers = []
        self.input = None
        self.output = None
        self.id = name
        self.optimizer_cls = None
        self.loss = None
        self.labels = None

    def _build(self, build_fn):
        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.variable_scope(str(self.id)):
                self.labels, self.optimizer_cls, self.loss = build_fn()

    def fit(self, x, y, lr, epochs, log_every=1):
        with self.graph.as_default():
            global_step = tf.contrib.framework.get_or_create_global_step()
            train_op = self.optimizer_cls(learning_rate=lr).minimize(self.loss,
                                                                     global_step=global_step)

            log_map = {'step': global_step, 'loss': self.loss}
            hooks = [tf.train.LoggingTensorHook(tensors=log_map, every_n_iter=log_every)]

            with tf.train.SingularMonitoredSession(hooks=hooks) as sess:
                for _ in range(epochs):
                    sess.run(train_op, feed_dict={self.input: x, self.labels: y})

    def add(self, l):
        '''
        Add this later to the model. We will take care of the unique naming
        '''
        if len(self.layers) == 0:
            self.input = l
        self.output = l
        self.layers.append(l)

    def __repr__(self):
        return '\n\n'.join([str(x) + ': ' + str(l) for x, l in enumerate(self.layers)])
