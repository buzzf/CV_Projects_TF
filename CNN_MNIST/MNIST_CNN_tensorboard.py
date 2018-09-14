import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('../learn/MNIST_DATA', one_hot=True)

batch_size = 100
n_batch = mnist.train.num_examples // batch_size

# 初始化权值
def weight_variable(shape,name):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial, name=name)


# 初始化偏置值
def bias_variable(shape, name):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial, name=name)


def variable_summaries(var):
	with tf.name_scope('summaries'):
		mean = tf.reduce_mean(var)
		tf.summary.scalar('mean', mean)
		with tf.name_scope('stddev'):
			stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
		tf.summary.scalar('stddev', stddev)
		tf.summary.scalar('max', tf.reduce_max(var))
		tf.summary.scalar('min', tf.reduce_min(var))
		tf.summary.histogram('histogram', var)

# conv layer
def conv2d(x, W):
	# x input tensor of shape [batch, in_height, in_width, in_channels]
	# W  filter / kernel tensor of shape [filter_height, filter_width, in_channels, out_channels], filter: Must have the same type as input
	# strides[0] = strides[3] = 1, means 0, 3位是没什么意义的占位符 , strides[1] means step of x方向,横向, strides[2] means step of y方向
	# padding: A string from: "SAME", "VALID", "SAME" 在外围补0
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# pooling layer
def max_pool_2x2(x):
	# ksize [1, x, y, 1], kernel size is x * y
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

with tf.name_scope('input'):
	x = tf.placeholder(tf.float32, [None, 784], name='x_input')
	y = tf.placeholder(tf.float32, [None, 10], name='y_input')
	with tf.name_scope('x_image'):
		# 改变x的格式转为4D的向量[batch, in_height, in_width, in_channels]
		x_image = tf.reshape(x, [-1, 28, 28, 1], name='x_image')



with tf.name_scope('conv1_layer'):
	with tf.name_scope('weights_conv1'):
		# 初始化第一个卷积层的权值和偏置值
		W_conv1 = weight_variable([5, 5, 1, 32], name='W_conv1')  # 5*5的采样窗口，32个卷积核从1个平面抽取特征, 即输出得到32个特征平面，因为是黑白图片，通道为1， 如果彩色则[5,5,3,32]
		variable_summaries(W_conv1)
	with tf.name_scope('biases_conv1'):
		b_conv1 = bias_variable([32], name='b_conv1')  # 每个卷积核一个偏置值
		variable_summaries(b_conv1)
	with tf.name_scope('conv1_fm'):
		conv1_fm = conv2d(x_image, W_conv1) + b_conv1
	with tf.name_scope('wx_plus_b_conv1'):
		# 把x_image和权值向量进行卷积，再加上偏置值，然后relu
		h_conv1 = tf.nn.relu(conv1_fm)
	with tf.name_scope('pooling_conv1'):
		h_pool1 = max_pool_2x2(h_conv1)


with tf.name_scope('conv2_layer'):
	with tf.name_scope('weights_conv2'):
	# 初始化第二个卷积层的权值和偏置值
		W_conv2 = weight_variable([5, 5, 32, 64], name='W_conv2')  # 5*5的采样窗口，64个卷积核从32个平面抽取特征
		variable_summaries(W_conv2)
	with tf.name_scope('bias_conv2'):
		b_conv2 = bias_variable([64], name='b_conv2')  # 每个卷积核一个偏置值
		variable_summaries(b_conv2)
	with tf.name_scope('features_conv2'):
		features_conv2 = conv2d(h_pool1, W_conv2) + b_conv2
	with tf.name_scope('relu'):
		h_conv2 = tf.nn.relu(features_conv2)
	with tf.name_scope('pooling_conv2'):
		h_pool2 = max_pool_2x2(h_conv2)
	with tf.name_scope('h_pool2_reshape'):
		# 把池化层2的输出扁平化为一维
		h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])


# 28*28的图片第一次卷积后还是28*28（SAME padding大小不变）， 第一次池化后变为14*14
# 第二次卷积后为14*14， 第二次池化后为7*7
# 通过上面的操作后得到64张7*7的平面

with tf.name_scope('fc1_layer'):
	with tf.name_scope('weight_fc1'):
		# 初始化第一个全连接层的权值
		W_fc1 = weight_variable([7*7*64, 1024], name='W_fc1')  # 上一层有&×&×64个神经元， 全连接层有1024个神经元
		variable_summaries(W_fc1)
	with tf.name_scope('bias_fc1'):
		b_fc1 = bias_variable([1024], name='b_fc1')  # 1024个节点
		variable_summaries(b_fc1)
	# with tf.name_scope('h_pool2_reshape'):
	# 	# 把池化层2的输出扁平化为一维
	# 	h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
	with tf.name_scope('wx_plus_b_fc1'):
		wx_plus_b_fc1 = tf.matmul(h_pool2_flat, W_fc1) + b_fc1
	with tf.name_scope('output_fc1'):
		# 第一个全连接层的输出
		h_fc1 = tf.nn.relu(wx_plus_b_fc1)
	with tf.name_scope('keep_prob'):
		# keep_prob dropout
		keep_prob = tf.placeholder(tf.float32)
	with tf.name_scope('dropout_fc1'):
		h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

with tf.name_scope('fc2_layer'):	
	with tf.name_scope('fc2_weight'):	
		# 初始化第二个全连接层
		W_fc2 = weight_variable([1024, 10], name='W_fc2')
		variable_summaries(W_fc2)
	with tf.name_scope('fc2_bias'):
		b_fc2 = bias_variable([10], name='b_fc2')
		variable_summaries(b_fc2)
	with tf.name_scope('wx_plus_b_fc2'):
		fc2_out = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
		# 计算输出
		# prediction = tf.nn.softmax(fc2_out)
		prediction = fc2_out    # softmax交叉熵已经包含了softmax操作


with tf.name_scope('loss'):
	cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
	tf.summary.scalar('loss', cross_entropy_loss)
with tf.name_scope('train'):
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy_loss)

with tf.name_scope('accuracy'):
	with tf.name_scope('correct_prediction'):
		correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
	with tf.name_scope('accuracy'):
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		tf.summary.scalar('accuracy', accuracy)


# merge all summary
merged = tf.summary.merge_all()


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter('logs/train', sess.graph)
    test_writer = tf.summary.FileWriter('logs/test', sess.graph)
    for i in range(1001):
    	# train models
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.7})
        # record train datasets params
        summary = sess.run(merged, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})
        train_writer.add_summary(summary, i)
        # record test datasets params
        batch_xs, batch_ys = mnist.test.next_batch(batch_size)
        summary = sess.run(merged, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})
        test_writer.add_summary(summary, i)

        if i%100==0:
        	test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
        	train_acc = sess.run(accuracy, feed_dict={x: mnist.train.images[:10000], y: mnist.train.labels[:10000], keep_prob: 1.0})
        	print('Iter {}, testing accuracy = {}, training accuracy = {}'.format(i, str(test_acc), str(train_acc)))
       
