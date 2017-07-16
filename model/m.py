#!/usr/bin/env python

import tensorflow as tf


class Model(object):
    '''
    Base Model for building tensorflow graphs.

    Example usage: TBD
    '''

    def __init__(self, name, lr=1e-3):
        # Model building
        self.layers = []
        self.input = None
        self.output = None
        self.id = name

        # Result from model building
        self.labels = None
        self.optimizer_cls = None
        self.loss = None

        # Training
        self.session = None
        self.train_op = None

        # learning rate
        self.lr = lr
        self.lrt = None

    def _build(self, build_fn):
        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.variable_scope(str(self.id)):
                self.labels, self.optimizer_cls, self.loss = build_fn()

    def set_lr(self, lr):
        self.lr = lr
        print("learning rate to: ", lr)

    def _get_session(self):
        if self.session is None:
            global_step = tf.contrib.framework.get_or_create_global_step()
            self.lrt = tf.placeholder(tf.float32, shape=[])
            self.train_op = self.optimizer_cls(learning_rate=self.lrt).minimize(self.loss,
                                                                                global_step=global_step)

            self.session = tf.Session()
            self.session.run(tf.global_variables_initializer())

        return self.session

    def fit(self, x, y, epochs, log_every=1):
        with self.graph.as_default():
            sess = self._get_session()
            for _ in range(epochs):
                l, _ = sess.run([self.loss, self.train_op], feed_dict={
                    self.input: x, self.labels: y, self.lrt: self.lr})
                print("Loss: ", l)

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

    def save(self, filename):
        with self.graph.as_default():
            saver = tf.train.Saver()
            saver.save(self._get_session(), filename)
