import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_DATA', one_hot=True)


batch_size = 100
n_batch = mnist.train.num_examples // batch_size

def variable_summaries(var):
	with tf.name_scope('summaries'):
		mean = tf.reduce_mean(var)
		tf.summary.scalar('mean', mean) # 平均值
		with tf.name_scope('stddev'):
			stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
		tf.summary.scalar('stddev', stddev)          # 标准差
		tf.summary.scalar('max', tf.reduce_max(var)) # 最大值
		tf.summary.scalar('min', tf.reduce_min(var)) # 最小值
		tf.summary.histogram('histogram', var)       # 直方图

# 命名空间
with tf.name_scope('input'):
	x = tf.placeholder(tf.float32, [None, 784], name='x-input')
	y = tf.placeholder(tf.float32, [None, 10], name='y-input')
	keep_prob = tf.placeholder(tf.float32)   # dropout param
	lr = tf.Variable(0.001, dtype=tf.float32)

# ******* create a simple N_net **************

with tf.name_scope('layer'):
	with tf.name_scope('wights'):
		w = tf.Variable(tf.zeros([784, 10]), name='w')
		variable_summaries(w)
	with tf.name_scope('biases'):
		b = tf.Variable(tf.zeros([10]), name='b')
		variable_summaries(b)
	with tf.name_scope('wx_plus_b'):
		wx_plus_b = tf.matmul(x, w) + b
	with tf.name_scope('softmax'):
		prediction = tf.nn.softmax(wx_plus_b)

# ****** two layers network  **********
'''
在正态分布的曲线中，横轴区间（μ-σ，μ+σ）内的面积为68.268949%。 
横轴区间（μ-2σ，μ+2σ）内的面积为95.449974%。 
横轴区间（μ-3σ，μ+3σ）内的面积为99.730020%。 
X落在（μ-3σ，μ+3σ）以外的概率小于千分之三，在实际问题中常认为相应的事件是不会发生的，基本上可以把区间（μ-3σ，μ+3σ）看作是随机变量X实际可能的取值区间，这称之为正态分布的“3σ”原则。 
在tf.truncated_normal中如果x的取值在区间（μ-2σ，μ+2σ）之外则重新进行选择。这样保证了生成的值都在均值附近。
'''
# w_l1 = tf.Variable(tf.truncated_normal([784, 50], stddev=0.1)) # 从截断的正态分布中输出随机值。
# # w_l1 = tf.Variable(tf.random_normal([784, 50], stddev=0.1)) 
# b_l1 = tf.Variable(tf.zeros([50]) + 0.1)
# L1 = tf.nn.relu(tf.matmul(x, w_l1) + b_l1)
# L1_drop = tf.nn.dropout(L1, keep_prob)

# w_l2 = tf.Variable(tf.random_normal([50, 10]))
# b_l2 = tf.Variable(tf.zeros([10]) + 0.1)
# prediction = tf.nn.softmax(tf.matmul(L1_drop, w_l2) + b_l2)


# *************** 代价函数 **************
# 二次代价函数
# loss = tf.reduce_mean(tf.square(y - prediction))
# 交叉熵代价函数
with tf.name_scope('loss'):
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
	tf.summary.scalar('loss', loss)

# ************* 优化器的使用 *****************

# train_step = tf.train.GradientDescentOptimizer(0.9).minimize(loss)   # 使用SGD梯度下降法
with tf.name_scope('train'):
	train_step = tf.train.AdamOptimizer(lr).minimize(loss)  

with tf.name_scope('accuracy'):
	with tf.name_scope('correct_prediction'):
		correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1)) # argmax返回一维张量中最大的值所在的index
	with tf.name_scope('accuracy'):
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) # tf.cast change a bool variable to a float32
		tf.summary.scalar('accuracy', accuracy)

# 合并所有的summary
merged = tf.summary.merge_all()


with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	writer = tf.summary.FileWriter('logs/', sess.graph)
	for epoch in range(51):
		sess.run(tf.assign(lr, 0.001 * (0.95 ** epoch)))  # 学习率不断减小
		for batch in range(n_batch):
			batch_xs, batch_ys = mnist.train.next_batch(batch_size)
			summary,_ = sess.run([merged, train_step], feed_dict={x: batch_xs, y:batch_ys, keep_prob: 1.0})

		writer.add_summary(summary, epoch)
		learning_rate = sess.run(lr)
		train_acc = sess.run(accuracy, feed_dict={x:mnist.train.images, y:mnist.train.labels, keep_prob: 1.0})
		test_acc = sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels, keep_prob: 1.0})
		print('Iter ' + str(epoch) + ', testing accuracy: ' + str(test_acc) + ', train accuracy: ' + str(train_acc))


