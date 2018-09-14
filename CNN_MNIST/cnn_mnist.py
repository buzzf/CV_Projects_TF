#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Author: zzf
# @Date:   2018-08-25 17:45:55
# @Last Modified by:   zzf
# @Last Modified time: 2018-08-25 18:52:16

import numpy as np 
import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data

# (55000 * 28 * 28) 55000张训练图片
mnist = input_data.read_data_sets('../learn/MNIST_DATA', one_hot=True)

# 表示张量（Tensor）的第一个维度可以使任何长度
input_x = tf.placeholder(tf.float32, [None, 28*28]) / 255. 
output_y = tf.placeholder(tf.float32, [None, 10])
input_x_images = tf.reshape(input_x, [-1, 28, 28 ,1])

test_x = mnist.test.images[:3000]
test_y = mnist.test.labels[:3000]


# 初始化权值
def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

# 初始化偏置值
def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)


# Neural Network layers
conv1 = tf.layers.conv2d(
		inputs=input_x_images,  # 形状 [28, 28, 1]
		filter=[],             # filters num, output depth 
		kernel_size=[5, 5],
		strides=1,
		padding='SAME',         # 补零方案 SAME or VALID, SAME表示输出大小不变，需要外围补零
		activation=tf.nn.relu 
		)                       # 形状 [28, 28, 32]

pool1 = tf.nn.max_pool(value, ksize, strides, padding) 