#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Author: zzf
# @Date:   2018-08-27 19:04:06
# @Last Modified by:   zzf
# @Last Modified time: 2018-08-27 19:29:50

import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('./MNIST_DATA', one_hot=True)

batch_size = 100
n_batch = mnist.train.num_examples // batch_size

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)   # dropout param
lr = tf.Variable(0.001, dtype=tf.float32)

# ******* create a simple N_net **************

# w = tf.Variable(tf.zeros([784, 10]))
# b = tf.Variable(tf.zeros([10]))
# prediction = tf.nn.softmax(tf.matmul(x, w) + b)
# prediction = tf.matmul(x, w) + b # 使用交叉熵代价函数时有softmax激活函数

# ****** mul layers network  **********
'''
在正态分布的曲线中，横轴区间（μ-σ，μ+σ）内的面积为68.268949%。 
横轴区间（μ-2σ，μ+2σ）内的面积为95.449974%。 
横轴区间（μ-3σ，μ+3σ）内的面积为99.730020%。 
X落在（μ-3σ，μ+3σ）以外的概率小于千分之三，在实际问题中常认为相应的事件是不会发生的，基本上可以把区间（μ-3σ，μ+3σ）看作是随机变量X实际可能的取值区间，这称之为正态分布的“3σ”原则。 
在tf.truncated_normal中如果x的取值在区间（μ-2σ，μ+2σ）之外则重新进行选择。这样保证了生成的值都在均值附近。
'''
w_l1 = tf.Variable(tf.truncated_normal([784, 2000], stddev=0.1)) # 从截断的正态分布中输出随机值。
# w_l1 = tf.Variable(tf.random_normal([784, 1000], stddev=0.1)) 
b_l1 = tf.Variable(tf.zeros([2000]) + 0.1)
L1 = tf.nn.relu(tf.matmul(x, w_l1) + b_l1)
L1_drop = tf.nn.dropout(L1, keep_prob)

w_l2 = tf.Variable(tf.truncated_normal([2000, 2000], stddev=0.1))
b_l2 = tf.Variable(tf.zeros([2000]) + 0.1)
L2 = tf.nn.relu(tf.matmul(L1_drop, w_l2) + b_l2)
L2_drop = tf.nn.dropout(L2, keep_prob)

w_l3 = tf.Variable(tf.truncated_normal([2000, 1000], stddev=0.1))
b_l3 = tf.Variable(tf.zeros([1000]) + 0.1)
L3 = tf.nn.relu(tf.matmul(L2_drop, w_l3) + b_l3)
L3_drop = tf.nn.dropout(L3, keep_prob)

w_l4 = tf.Variable(tf.truncated_normal([1000, 10], stddev=0.1))
b_l4 = tf.Variable(tf.zeros([10]) + 0.1)
# prediction = tf.nn.softmax(tf.matmul(L3_drop, w_l4) + b_l4)
prediction = tf.matmul(L3_drop, w_l4) + b_l4 # 使用交叉熵代价函数时有softmax激活函数


# *************** 代价函数 **************
# 二次代价函数
# loss = tf.reduce_mean(tf.square(y - prediction))

# 交叉熵代价函数
# tensorflow交叉熵计算函数输入中的logits都不是softmax或sigmoid的输出，而是softmax或sigmoid函数的输入，因为它在函数内部进行sigmoid或softmax操作
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))


# ************* 优化器的使用 *****************

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)   # 使用SGD梯度下降法
# train_step = tf.train.AdamOptimizer(lr).minimize(loss)  
# train_step = tf.train.AdadeltaOptimizer(1., 0.95, 1e-6).minimize(loss)
# train_step = tf.train.MomentumOptimizer(0.01, 0.9).minimize(loss)


correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1)) # argmax返回一维张量中最大的值所在的index

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) # tf.cast change a bool variable to a float32

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for epoch in range(21):
		# sess.run(tf.assign(lr, 0.001 * (0.95 ** epoch)))  # 学习率不断减小
		for batch in range(n_batch):
			batch_xs, batch_ys = mnist.train.next_batch(batch_size)
			sess.run(train_step, feed_dict={x: batch_xs, y:batch_ys, keep_prob: 0.85})

		# learning_rate = sess.run(lr)
		train_acc = sess.run(accuracy, feed_dict={x:mnist.train.images, y:mnist.train.labels, keep_prob: 1.0})
		test_acc = sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels, keep_prob: 1.0})
		print('Iter ' + str(epoch) + ', testing accuracy: ' + str(test_acc) + ', train accuracy: ' + str(train_acc))


