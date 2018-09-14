#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Author: zzf
# @Date:   2018-09-11 14:44:28
# @Last Modified by:   zzf
# @Last Modified time: 2018-09-12 14:38:44

import tensorflow as tf 


'''
# 情景一：

c = tf.constant(4.0)

with tf.Session() as sess:
	c_out = sess.run(c)
	print(c_out)
	assert c.graph == tf.get_default_graph()
	print(c.graph)
	print(tf.get_default_graph())


# 情景二

g = tf.Graph()
with g.as_default():
	c = tf.constant(3.0)

	with tf.Session(graph=g) as sess:
		c_out = sess.run(c)
		print(c_out)
		print(g)
		print(tf.get_default_graph())
'''

# 情景三

c=tf.constant(value=1)
print('c: \n', c.graph)
print('default: \n', tf.get_default_graph())

g1 = tf.Graph()
print('g1: \n', g1)
with g1.as_default():
    c1 = tf.constant(4.0)
    print('c1: \n', c1.graph)
 
g2 = tf.Graph()
print('g2: \n', g2)
with g2.as_default():
	c2 = tf.constant(20.0)
	print('c2: \n', c2.graph)

e=tf.constant(value=15)
print('e: \n', e.graph)

with tf.Session(graph=g1) as sess1:
    print(sess1.run(c1))
with tf.Session(graph=g2) as sess2:
    print(sess2.run(c2))

"""
上面的例子里面创创建了一个新的图g1，然后把g1设为默认，那么接下来的操作不是在默认的图中，而是在g1中了。你也可以认为现在g1这个图就是新的默认的图了。 然后又创建新的图g2，做同样的操作。
要注意的是，最后一个量e不是定义在with语句里面的，也就是说，e会包含在最开始的那个图中。也就是说，要在某个graph里面定义量，要在with语句的范围里面定义。
从结果的graph ID可以验证以上说法

"""