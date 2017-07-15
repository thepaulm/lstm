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
        self.session = None
        self.train_op = None
        self.global_step = None
        self.once = False

    def _build(self, build_fn):
        # self.graph = tf.Graph()
        # with self.graph.as_default():
        if True:
            with tf.variable_scope(str(self.id)):
                self.labels, self.optimizer_cls, self.loss = build_fn()

    def set_lr(self, lr):
        self.global_step = tf.contrib.framework.get_or_create_global_step()
        self.train_op = self.optimizer_cls(learning_rate=lr).minimize(self.loss,
                                                                      global_step=self.global_step)

        #
        # from: https://stackoverflow.com/questions/33919948/how-to-set-adaptive-learning-rate-for-gradientdescentoptimizer
        #
        # learning_rate = tf.placeholder(tf.float32, shape=[])
        # # ...
        # train_step = tf.train.GradientDescentOptimizer(
        # learning_rate=learning_rate).minimize(mse)
        #
        # sess = tf.Session()
        #
        # # Feed different values for learning rate to each training step.
        # sess.run(train_step, feed_dict={learning_rate: 0.1})
        # sess.run(train_step, feed_dict={learning_rate: 0.1})
        # sess.run(train_step, feed_dict={learning_rate: 0.01})
        # sess.run(train_step, feed_dict={learning_rate: 0.01})

    def _get_session(self):
        if self.session is None:
            self.session = tf.Session()
            if not self.once:
                self.session.run(tf.global_variables_initializer())
                self.once = True
            else:
                tf.variables_initializer(self.train_op)

        return self.session

    def fit(self, x, y, epochs, log_every=1):
        # with self.graph.as_default():
        if True:
            # log_map = {'step': global_step, 'loss': self.loss}
            # hooks = [tf.train.LoggingTensorHook(tensors=log_map, every_n_iter=log_every)]

            # with tf.train.SingularMonitoredSession(hooks=hooks) as sess:
            sess = self._get_session()
            for _ in range(epochs):
                l, _ = sess.run([self.loss, self.train_op], feed_dict={
                                self.input: x, self.labels: y})
                print("Loss: ", l)

            # self.session.close()
            # self.session = None

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
