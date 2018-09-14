import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 

x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]
noise = np.random.normal(0, 0.02, x_data.shape)
y_data = np.square(x_data) + noise

x = tf.placeholder(tf.float32, [None, 1]) # [None, 1] means x.shape ,row could be any num, column is 1
y = tf.placeholder(tf.float32, [None, 1])

# 定义神经网络的中间层

weights_l1 = tf.Variable(tf.random_normal([1, 10]))
biases_l1 = tf.Variable(tf.zeros([1, 10]))
wx_plus_b_l1 = tf.matmul(x, weights_l1) + biases_l1
L1 = tf.nn.tanh(wx_plus_b_l1) # activate function

# 定义神经网络的输出层
weights_l2 = tf.Variable(tf.random_normal([10, 1]))
biases_l2 = tf.Variable(tf.zeros([1, 1]))
wx_plus_b_l2 = tf.matmul(L1, weights_l2) + biases_l2
prediction = tf.nn.tanh(wx_plus_b_l2)


# 二次代价函数

loss = tf.reduce_mean(tf.square(y - prediction))
train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
 
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for i in range(2000):
		sess.run(train, feed_dict={x:x_data, y:y_data})

	# get predict number
	prediction_value = sess.run(prediction, feed_dict={x:x_data})
	plt.figure()
	plt.scatter(x_data, y_data)
	plt.plot(x_data, prediction_value, 'r-', lw=5)
	plt.savefig('./linear_regression.png')
	plt.show()
