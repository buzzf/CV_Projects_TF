import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('../learn/MNIST_DATA', one_hot=True)

# 输入图片是28*28
n_inputs = 28
max_time = 28
lstm_size = 100 # 隐层单元， 100个block
n_classes = 10 # 10个分类
batch_size = 50
n_batch = mnist.train.num_examples // batch_size


x = tf.placeholder(tf.float32, [None, 784])   # 50 * 784
y = tf.placeholder(tf.float32, [None, 10])    # 50 * 10

weights = tf.Variable(tf.truncated_normal([lstm_size, n_classes], stddev=0.1))
biases = tf.Variable(tf.constant(0.1, shape=[n_classes]))

def RNN(X, weights, biases):
	# inputs = [batch_size, max_time, n_inputs]
	inputs = tf.reshape(X, [-1, max_time, n_inputs])  # 50 * 28 *28
	# 定义LSTM基本cell, 中间隐藏层的block
	lstm_cell = tf.contrib.rnn.BasicLSTMCell(lstm_size)

	# 对于tf.nn.dynamic_rnn，有
	# outputs, The RNN output Tensor
	# If time_major == False (default), this will be a Tensor shaped: 
	# 			[batch_size, max_time, cell.output_size].
	# If time_major == True, this will be a Tensor shaped: 
	# 			[max_time, batch_size, cell.output_size].
	# final_state,  是由(c,h)组成的tuple，均为[batch_size, cell.state_size]格式
	# final_state[0] 是 cell state
	# final_state[1] 是 hidden state

	# hidden state == max_time取最后一个时间序列时cell.output_size
	
	outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, inputs, dtype=tf.float32)

	results = tf.nn.softmax(tf.matmul(final_state[1], weights) + biases)
	return results


#计算RNN的返回结果
prediction= RNN(x, weights, biases)  
#损失函数
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
#使用AdamOptimizer进行优化
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#结果存放在一个布尔型列表中
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1)) # argmax返回一维张量中最大的值所在的位置
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32)) # 把correct_prediction变为float32类型
init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	for epoch in range(6):
		for batch in range(n_batch):
			batch_xs,batch_ys =  mnist.train.next_batch(batch_size)
			sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys})

		acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
		print ("Iter " + str(epoch) + ", Testing Accuracy= " + str(acc))