from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
from scipy.misc import imread, imresize, imsave
from os.path import join
import manage_images
import DCGAN
import cv2
import os
import time
TRAIN_LOG_DIR = os.path.join('Log/train/', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
TRAIN_CHECK_POINT = 'check_point/'+time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
BATCH_SIZE = 128
IMG_HEIGHT = manage_images.IMG_HEIGHT
IMG_WIDTH = manage_images.IMG_WIDTH
IMG_CHANNELS = manage_images.IMG_CHANNELS
NOISE_SIZE = 100
learning_rate = 0.0002
beta1 = 0.5
epoch_num = 2000
image_save_path = os.path.join('g_images/',time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
if not tf.gfile.Exists(TRAIN_LOG_DIR):
    tf.gfile.MakeDirs(TRAIN_LOG_DIR)
if not tf.gfile.Exists(image_save_path):
    tf.gfile.MakeDirs(image_save_path)
if not tf.gfile.Exists(TRAIN_CHECK_POINT):
    tf.gfile.MakeDirs(TRAIN_CHECK_POINT)

batch_image_set, images_num = manage_images.data_processing(batch_size=BATCH_SIZE)

with tf.Graph().as_default():
    real_input = tf.placeholder(tf.float32, [None, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS])
    noise_input = tf.placeholder(tf.float32, [None, NOISE_SIZE])
	#One step G, one step D for real images, and one step D for fake images from G.
    G_logits = DCGAN.generator(noise_input)
    r_D_logits, r_D = DCGAN.discriminator(real_input, reuse=False)
    f_D_logits, f_D = DCGAN.discriminator(G_logits, reuse=True)
    
    D_r_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=r_D_logits,labels=tf.ones_like(r_D)))
    D_f_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_D_logits,labels=tf.zeros_like(f_D)))
    with tf.name_scope('loss'):
       D_loss = D_r_loss + D_f_loss
       tf.summary.scalar('D_loss', D_loss)
       G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_D_logits,labels=tf.ones_like(f_D)))
       tf.summary.scalar('G_loss', G_loss)
    t_vars = tf.trainable_variables()

    D_vars = [var for var in t_vars if 'discriminator' in var.name]
    print(len(D_vars))                   
    G_vars = [var for var in t_vars if 'generator' in var.name]
    print(len(G_vars))
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        d_optim = tf.train.AdamOptimizer(learning_rate, beta1=beta1) \
            .minimize(D_loss, var_list=D_vars)
        g_optim = tf.train.AdamOptimizer(learning_rate, beta1=beta1) \
            .minimize(G_loss, var_list=G_vars)
    merged = tf.summary.merge_all()
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        train_writer = tf.summary.FileWriter(TRAIN_LOG_DIR, sess.graph)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        step = 0
        for epoch in range(epoch_num):
            
            D_fake_loss = 0
            D_real_loss = 0
            t_G_loss = 0
            for i in range(images_num // BATCH_SIZE):
                batch_noise = np.random.uniform(-1, 1, [BATCH_SIZE, NOISE_SIZE]).astype(np.float32)
                # Run D once and G twice. You can adjust it for your own data sets.
                sess.run(d_optim, feed_dict={real_input: batch_image_set[i], noise_input: batch_noise})
                
                sess.run(g_optim, feed_dict={real_input: batch_image_set[i], noise_input: batch_noise})
                sess.run(g_optim, feed_dict={real_input: batch_image_set[i], noise_input: batch_noise})

                batch_D_fake_loss = sess.run(D_f_loss, feed_dict={noise_input: batch_noise})
                batch_D_real_loss = sess.run(D_r_loss, feed_dict={real_input: batch_image_set[i]})
                sammury, batch_G_loss = sess.run([merged, G_loss], feed_dict={real_input: batch_image_set[i], noise_input: batch_noise})

                D_fake_loss += batch_D_fake_loss
                D_real_loss += batch_D_real_loss
                t_G_loss += batch_G_loss
                step += 1
                if i % 10 == 0:
                   train_writer.add_summary(summary, step)
                   print('Epoch %d; Batch %d: D_fake_loss is %.5f; D_real_loss is %.5f; G_loss is %.5f.'%(epoch+1, i, batch_D_fake_loss,
                                                                                           batch_D_real_loss, \
                                                                                           batch_G_loss))
            D_fake_loss = D_fake_loss / (images_num // BATCH_SIZE)
            D_real_loss = D_real_loss / (images_num // BATCH_SIZE)
            t_G_loss = t_G_loss / (images_num // BATCH_SIZE)
            print('Epoch %d: D_fake_loss is %.5f; D_real_loss is %.5f; G_loss is %.5f.'%(epoch+1, D_fake_loss,
                                                                                       D_real_loss,
                                                                                       t_G_loss))
            #Generate images and merge them together.
            batch_noise = np.random.uniform(-1, 1, [BATCH_SIZE, NOISE_SIZE]).astype(np.float32)
            g_images = G_logits.eval({noise_input:batch_noise})
            image = manage_images.merge_images(g_images)
            imsave(join(image_save_path, 'epoch' + str(epoch + 1) + '.jpg'), image)
            saver.save(sess, TRAIN_CHECK_POINT + 'train_model.ckpt', global_step=epoch)
