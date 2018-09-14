#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Author: zzf
# @Date:   2018-08-24 14:36:03
# @Last Modified by:   zzf
# @Last Modified time: 2018-08-24 14:47:19

import tensorflow as tf 

# y = W * x + b

W = tf.Variable(2.0, dtype=tf.float32, name='Weight')
b = tf.Variable(5.0, dtype=tf.float32, name='bias')
x = tf.placeholder(dtype=tf.float32, name='input')

with tf.name_scope('Output'):   # 输出命名空间
	y = W * x + b

path = './log'

# 创建所有变量的初始化的操作
init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	writer = tf.summary.FileWriter(path, sess.graph)
	result = sess.run(y, {x: 3.0})
	print('y = {}'.format(result))


