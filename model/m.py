#!/usr/bin/env python

import tensorflow as tf


class Model(object):
    '''
    Base Model for building tensorflow graphs.

    Example usage: TBD
    '''
    def __init__(self):
        self.layers = []
        self.input = None
        self.output = None
        self.id = hex(id(self))

    def _build(self, build_fn):
        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.variable_scope(str(self.id)):
                build_fn()

    def add(self, l):
        '''
        Add this later to the model. We will take care of the unique naming
        '''
        if len(self.layers) == 0:
            self.input = l
        self.output = l
        self.layers.append(l)

    def name(self, n):
        return str(self.id) + '_' + n

    def __repr__(self):
        return '\n\n'.join([str(x) + ': ' + str(l) for x, l in enumerate(self.layers)])
