#!/usr/bin/env python


class Model(object):
    '''
    Base Model for building tensorflow graphs.

    Example usage:
    class AModel(Model):
        def __init__(self):
            super().__init__()
            with tf.variable_scope(self.name('A_input')):
                self.add(tf.placeholder(tf.float32, (None, ) + img_size + (3,)))

            with tf.variable_scope(self.name('A_conv_1')):
                self.add(tf.layers.conv2d(self.output, 64, (2, 2), strides=(2, 2),
                         padding='valid', activation=tf.nn.relu))
                self.add(tf.nn.max_pool(self.output, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                        padding='VALID'))

    '''
    def __init__(self):
        self.layers = []
        self.input = None
        self.output = None
        self.id = hex(id(self))

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
