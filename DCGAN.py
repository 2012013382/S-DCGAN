from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
#Channels of this network is a little different from their in the paper.
'''
Discriminator network
   Input: [BATCH_SIZE, 64, 64 ,3]
   conv1: [BATCH_SIZE, 32, 32, 64]
   conv2: [BATCH_SIZE, 16, 16, 128]
   conv3: [BATCH_SIZE, 8, 8, 256]
   conv4: [BATCH_SIZE, 4, 4, 512]
   reshape: [BATCH_SIZE, 8192]
   fc5: [BATCH_SIZE, 1]   
'''
'''
Generator network
   Input: [BATCH_SIZE, 100]
   fc1: [BATCH_SIZE, 8192]
   reshape: [BATCH_SIZE, 4, 4, 512]
   conv_tran1: [BATCH_SIZE, 8, 8, 256]
   conv_tran2: [BATCH_SIZE, 16, 16, 128]
   conv_tran3: [BATCH_SIZE, 32, 32, 64]
   conv_tran4: [BATCH_SIZE, 64, 64, 3]
'''
#Leaky relu with leaky 0.2.
def leaky_relu(x, lk = 0.2):
    return tf.maximum(x, x * lk)
  
def _batch_norm(net, scope_name, training_state):
   net = slim.batch_norm(net, decay=0.9, epsilon=1e-5, scale=True, scope=scope_name, is_training=training_state)
   return net

#Define discriminator
def discriminator(inputs, reuse=False, training_state=True):#Default input shape[batch_size, 64, 64, 3]
    with tf.variable_scope('discriminator') as scope:
        if reuse:
           scope.reuse_variables()
        with slim.arg_scope([slim.conv2d], padding='SAME',
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                            weights_regularizer=slim.l2_regularizer(0.0005),
                            activation_fn=leaky_relu,
                            kernel_size=[5, 5],
                            stride=2):
            net = slim.conv2d(inputs, 64, scope='d_conv1')
            net = _batch_norm(net, scope='d_bd1', training_state)
            net = slim.conv2d(net, 64 * 2, scope='d_conv2')
            net = _batch_norm(net, scope='d_bd2', training_state)
            net = slim.conv2d(net, 64 * 4, scope='d_conv3')
            net = _batch_norm(net, scope='d_bd3', training_state)
            net = slim.conv2d(net, 64 * 8, scope='d_conv4')
            net = slim.fully_connected(tf.reshape(net,[-1, 8192]), 1, activation_fn=None, scope='d_fc5')
    return net, tf.nn.sigmoid(net)

#Define generator
def generator(z, training_state=True):#Default shape of z is [batch_size, 100]
    with tf.variable_scope('generator') as scope:
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], padding='SAME',
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                            weights_regularizer=slim.l2_regularizer(0.0005),
                            activation_fn=leaky_relu,
                            kernel_size=[5, 5],
                            stride=2):
            net = slim.fully_connected( z,
                                        8192,
                                        weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                        weights_regularizer=slim.l2_regularizer(0.0005),
                                        activation_fn=None,
                                        scope='g_fc1')
            net = tf.reshape(net, [-1, 4, 4, 64 * 8])
            net = tf.nn.relu(_batch_norm(net, 'g_bd1', training_state))
            net = slim.conv2d_transpose(net, num_outputs=64 * 4, activation_fn=None, scope='g_conv_tran1')
            net = tf.nn.relu(_batch_norm(net, 'g_bd2', training_state))
            net = slim.conv2d_transpose(net, num_outputs=64 * 2, activation_fn=None, scope='g_conv_tran2')
            net = tf.nn.relu(_batch_norm(net, 'g_bd3', training_state))
            net = slim.conv2d_transpose(net, num_outputs=64 * 1, activation_fn=None, scope='g_conv_tran4')
            net = tf.nn.relu(_batch_norm(net, 'g_bd4', training_state))
            net = slim.conv2d_transpose(net, num_outputs=3, activation_fn=None, scope='g_conv_tran5')
            net = tf.nn.tanh(net)
            
    return net


