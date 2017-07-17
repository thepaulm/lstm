#!/usr/bin/env python

import tensorflow as tf


class Model(object):
    '''
    Base Model for building tensorflow graphs.

    Example usage: TBD
    '''

    def __init__(self, name, tensorboard_dir=None, lr=1e-3):
        # Model building
        self.layers = []
        self.input = None
        self.output = None
        self.id = name

        # Result from model building
        self.labels = None
        self.optimizer_cls = None
        self.loss = None
        self.prediction = None

        # Training
        self.session = None
        self.train_op = None

        # learning rate
        self.lr = lr
        self.lrt = None

        # if you want tensorboard we put it here
        self.tensorboard_dir = tensorboard_dir
        self.fit_iterations = 0

    def _build(self, build_fn):
        '''
        _build should be called from subclass to build the model. Use add() method to
        add layers. return labels, prediction, optimizer class, loss.
        '''
        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.variable_scope(str(self.id)):
                self.labels, self.prediction, self.optimizer_cls, self.loss = build_fn()
                if self.tensorboard_dir:
                    tf.summary.scalar('loss', self.loss)
                    self.merged = tf.summary.merge_all()
                    self.tb_writer = tf.summary.FileWriter(self.tensorboard_dir, self.graph)

    def set_lr(self, lr):
        '''set_lr to update learning rate. call this at least once.'''
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
        '''fit to fix predictions to labels.'''
        with self.graph.as_default():
            sess = self._get_session()
            for _ in range(epochs):
                ops = [self.loss, self.train_op]
                if self.tensorboard_dir:
                    ops.append(self.merged)
                r = sess.run(ops, feed_dict={
                    self.input: x, self.labels: y, self.lrt: self.lr})
                self.fit_iterations += 1
                print("Loss: ", r[0])
                if self.tensorboard_dir:
                    self.tb_writer.add_summary(r[2], self.fit_iterations)

    def predict(self, x):
        '''predict to make prediction from observation.'''
        with self.graph.as_default():
            sess = self._get_session()
            res = sess.run(self.prediction, feed_dict={self.input: x})
            return res

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
        '''save graph'''
        with self.graph.as_default():
            saver = tf.train.Saver()
            saver.save(self._get_session(), filename)

    def load(self, filename):
        '''load graph'''
        with self.graph.as_default():
            saver = tf.train.Saver()
            saver.restore(self._get_session(), filename)
