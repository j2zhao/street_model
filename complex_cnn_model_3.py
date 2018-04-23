# Some code was borrowed from https://github.com/petewarden/tensorflow_makefile/blob/master/tensorflow/models/image/mnist/convolutional.py
import tensorflow as tf
import numpy as np
from enum import Enum
import utility

def raise_Error():
    raise ValueError("Unsupported Type")

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
# type = Data
def CNN(inputs, type):
    
    input_a = tf.cond(tf.equal(type, utility.Data.CUSTOM.value), 
        lambda: tf.reshape(inputs, [-1, 32, 32, 1]), lambda: tf.zeros([1, 32, 32, 1], tf.float32))
    input_b = tf.cond(tf.equal(type, utility.Data.MNIST.value), 
        lambda: tf.reshape(inputs, [-1, 28, 28, 1]), lambda: tf.zeros([1, 28, 28, 1], tf.float32))
    input_c = tf.cond(tf.equal(type, utility.Data.STREET.value), 
        lambda: tf.reshape(inputs, [-1, 32, 32, 3]), lambda: tf.zeros([1, 32, 32, 3], tf.float32))
    
    with tf.variable_scope('conv1a') as scope:
        weights = create_weights(shape=[5, 5, 1, 32])
        biases = create_biases([32])
        layer = tf.nn.conv2d(input= input_a,
                    filter= weights,
                    strides=[1, 1, 1, 1],
                    padding='SAME')
        layer = tf.nn.bias_add(layer, biases)
        conv1a = tf.nn.relu(layer, name= scope.name)
    with tf.variable_scope('conv1b') as scope:
        weights = create_weights(shape=[5, 5, 1, 32])
        biases = create_biases([32])
        paddings = tf.constant([[0, 0], [2, 2], [2, 2], [0, 0]])
        input_b = tf.pad(input_b, paddings)
        layer = tf.nn.conv2d(input= input_b,
                    filter= weights,
                    strides=[1, 1, 1, 1],
                    padding='SAME')
        layer = tf.nn.bias_add(layer, biases)
        conv1b = tf.nn.relu(layer, name= scope.name)
        
    with tf.variable_scope('conv1c') as scope:
        weights = create_weights(shape=[5, 5, 3, 32])
        biases = create_biases([32])
        layer = tf.nn.conv2d(input= input_c,
                    filter= weights,
                    strides=[1, 1, 1, 1],
                    padding='SAME')
        layer = tf.nn.bias_add(layer, biases)
        conv1c = tf.nn.relu(layer, name= scope.name)
        

    cases = [(tf.equal(type, utility.Data.CUSTOM.value), lambda: conv1a), (tf.equal(type, utility.Data.STREET.value), lambda: conv1c), 
            (tf.equal(type, utility.Data.MNIST.value), lambda: conv1b)]
    conv1 = tf.case(cases, lambda:conv1a)
    conv1.set_shape([None, 32, 32, 32])
    with tf.variable_scope('conv2') as scope:
        weights = create_weights(shape=[5, 5, 32, 32])
        biases = create_biases([32])
        layer = tf.nn.conv2d(input=conv1,
                     filter= weights,
                     strides=[1, 1, 1, 1],
                     padding='SAME')
        layer = tf.nn.bias_add(layer, biases)
        conv2 = tf.nn.relu(layer, name=scope.name)

    with tf.variable_scope('pool2') as scope:
        pool2 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
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
    
    flatten4_a = tf.cond(tf.equal(type, utility.Data.CUSTOM.value), 
        lambda: flatten4, lambda: tf.zeros([1, num_features], tf.float32))
    flatten4_b = tf.cond(tf.equal(type, utility.Data.MNIST.value), 
        lambda: flatten4, lambda: tf.zeros([1, num_features], tf.float32))
    flatten4_c = tf.cond(tf.equal(type, utility.Data.STREET.value), 
        lambda: flatten4, lambda: tf.zeros([1, num_features], tf.float32))

    with tf.variable_scope('fc4a') as scope:
        weights = create_weights(shape=[num_features, 1024])
        biases = create_biases([1024])
        fc4a = tf.matmul(flatten4_a, weights) + biases

    with tf.variable_scope('fc5a') as scope:
        weights = create_weights(shape=[1024, 10])
        biases = create_biases([10])
        fc5a = tf.matmul(fc4a, weights) + biases
    
    with tf.variable_scope('fc4a') as scope:
        weights = create_weights(shape=[num_features, 1024])
        biases = create_biases([1024])
        fc4b = tf.matmul(flatten4_b, weights) + biases

    with tf.variable_scope('fc5a') as scope:
        weights = create_weights(shape=[1024, 10])
        biases = create_biases([10])
        fc5b = tf.matmul(fc4b, weights) + biases

    with tf.variable_scope('fc4a') as scope:
        weights = create_weights(shape=[num_features, 1024])
        biases = create_biases([1024])
        fc4c = tf.matmul(flatten4_c, weights) + biases

    with tf.variable_scope('fc5a') as scope:
        weights = create_weights(shape=[1024, 10])
        biases = create_biases([10])
        fc5c = tf.matmul(fc4c, weights) + biases

    cases = [(tf.equal(type, utility.Data.CUSTOM.value), lambda: fc5a), (tf.equal(type, utility.Data.STREET.value), lambda: fc5c), 
            (tf.equal(type, utility.Data.MNIST.value), lambda: fc5b)]
    outputs = tf.case(cases, lambda:fc5c)
    return outputs

def loss(logits, labels):
    loss = tf.nn.softmax_cross_entropy_with_logits(
                logits=logits,
                labels=labels)
    loss = tf.reshape(loss, [-1])
    loss = tf.reduce_sum(loss)
    tf.summary.scalar('loss', loss)
    return loss