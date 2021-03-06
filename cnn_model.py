# Some code was borrowed from https://github.com/petewarden/tensorflow_makefile/blob/master/tensorflow/models/image/mnist/convolutional.py
import tensorflow as tf
import numpy as np
from enum import Enum

def create_weights(shape):
    '''Create a convolution filter variable with the specified name and shape,
    and initialize it using Xavier initialition.'''
    initializer = tf.contrib.layers.xavier_initializer_conv2d()
    variable = tf.Variable(initializer(shape=shape))
    return variable

def create_biases(shape):
    '''Create a bias variable with the specified name and shape and initialize
    it to zero.'''
    initializer = tf.constant_initializer(value=0.0, dtype=tf.float32)
    return tf.Variable(initializer(shape=shape))

def create_flatten_layer(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer = tf.reshape(layer, [-1, num_features])
    return layer

# Create model CNN
# type = 
def CNN(inputs, is_training=True):
    #inputs = tf.reshape(inputs, [-1, 32, 32, 3])
    inputs = tf.reshape(inputs, [-1, 28, 28, 3])
    paddings = tf.constant([[0, 0], [2, 2], [2, 2], [0, 0]])
    inputs = tf.pad(inputs, paddings)
    with tf.variable_scope('conv1a') as scope:
        weights = create_weights(shape=[5, 5, 3, 32])
        biases = create_biases([32])
        layer = tf.nn.conv2d(input=inputs,
                     filter= weights,
                     strides=[1, 1, 1, 1],
                     padding='SAME')
        layer = tf.nn.bias_add(layer, biases)
        conv1 = tf.nn.relu(layer, name= scope.name)

    with tf.variable_scope('conv2') as scope:
        weights = create_weights(shape=[5, 5, 32, 32])
        biases = create_biases([32])
        layer = tf.nn.conv2d(input=conv1,
                     filter=weights,
                     strides=[1, 1, 1, 1],
                     padding='SAME')
        layer = tf.nn.bias_add(layer, biases)
        conv2 = tf.nn.relu(layer, name=scope.name)

    with tf.variable_scope('pool2') as scope:
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name=scope.name)

    with tf.variable_scope('conv3') as scope:
        weights = create_weights(shape=[5, 5, 32, 64])
        biases = create_biases([64])
        layer = tf.nn.conv2d(input=pool2,
                     filter=weights,
                     strides=[1, 1, 1, 1],
                     padding='SAME')
        layer = tf.nn.bias_add(layer, biases)
        conv3 = tf.nn.relu(layer, name=scope.name)

    with tf.variable_scope('pool3') as scope:
        pool3 = tf.nn.max_pool(conv3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name=scope.name)
    
    with tf.variable_scope('flatten4') as scope:
        layer_shape = pool3.get_shape()
        num_features = layer_shape[1:].num_elements()
        flatten4 = tf.reshape(pool3, [-1, num_features], name=scope.name)
    
    with tf.variable_scope('fc4') as scope:
        weights = create_weights(shape=[num_features, 1024])
        biases = create_biases([1024])
        fc4 = tf.matmul(flatten4, weights) + biases

    with tf.variable_scope('fc5') as scope:
        weights = create_weights(shape=[1024, 10])
        biases = create_biases([10])
        outputs = tf.matmul(fc4, weights) + biases
    return outputs

def loss(logits, labels):
    loss = tf.nn.softmax_cross_entropy_with_logits(
                logits=logits,
                labels=labels)
    loss = tf.reshape(loss, [-1])
    loss = tf.reduce_sum(loss)
    tf.summary.scalar('loss', loss)
    return loss