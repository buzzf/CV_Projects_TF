import tensorflow as tf 
import numpy as np

'''
w = tf.Variable([[0.5, 1]])
x = tf.Variable([[2.0], [1.0]])
y = tf.matmul(w, x)
norm = tf.random_normal([2,3], mean=-1, stddev=4)
c = tf.constant([[1,2], [3,4], [5,6]])
shuff = tf.random_shuffle(c)

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init_op)
	print(y.eval())
	print(sess.run(norm))
	print(sess.run(shuff))

#*************** 常量 ******************

m1 = tf.constant([[3, 3]])  # 常量
m2 = tf.constant([[2], [3]])

product = tf.matmul(m1, m2)
with tf.Session() as sess:
	result = sess.run(product)
	print(result)

#*************** 变量 ******************

x = tf.Variable([1, 2])  # 变量  创建一个变量，初始化为[1,2]
a = tf.constant([3, 3])
sub = tf.subtract(x, a)
add = tf.add(x, a)

init = tf.global_variables_initializer() # 有变量是一定要先初始化

with tf.Session() as sess:
	sess.run(init)  
	print(sess.run(sub))
	print(sess.run(add))


state = tf.Variable(0, name='counter')
new_value = tf.add(state, 1)
update = tf.assign (state, new_value)  # 赋值op

init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)
	print(sess.run(state))
	for i in range(5):
		sess.run(update)
		print(sess.run(state))

'''


#*************** fetch feed ******************

# fetch 
'''
input1 = tf.constant(4.0)
input2 = tf.constant(2.0)
input3 = tf.constant(3.0)

add = tf.add(input2, input3)
mul = tf.multiply(input1, add)

with tf.Session() as sess:
	result = sess.run([mul, add])   # fetch 可以同时运行多个op
	print(result)


# feed

input1 = tf.placeholder(tf.float32)  # 创建占位符
input2 = tf.placeholder(tf.float32) 
output = tf.multiply(input1, input2)

with tf.Session() as sess:
	print(sess.run(output, feed_dict={input1: [7.0], input2:[2.0]})) # feed的数据以字典的形式传入
'''

#*************** simple example ******************

x_data = np.random.rand(100) # 100 random pot
y_data = x_data*0.1 + 0.2

# make a linear model
b = tf.Variable(0.)  
k = tf.Variable(0.)
y = k*x_data + b

# 二次代价函数
loss = tf.reduce_mean(tf.square(y_data - y))
# 定义一个用梯度下降法来进行训练的优化器
optimizer = tf.train.GradientDescentOptimizer(0.2)
# 最小化代价函数
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)
	for step in range(501):
		sess.run(train)
		if step%20==0:
			print(step, sess.run([k,b]))