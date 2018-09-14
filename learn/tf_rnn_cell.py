#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Author: zzf
# @Date:   2018-09-12 14:38:56
# @Last Modified by:   zzf
# @Last Modified time: 2018-09-13 15:35:53

import tensorflow as tf
import numpy as np

'''
前者是隐层的大小，后者是输出的大小。比如我们通常是将一个batch送入模型计算，设输入数据的形状为(batch_size, input_size)，那么计算时得到的隐层状态就是(batch_size, state_size)，输出就是(batch_size, output_size)。

'''

cell = tf.nn.rnn_cell.BasicRNNCell(num_units=128) # state_size = 128
print(cell.state_size) # 128

inputs = tf.placeholder(np.float32, shape=(32, 100)) # 32 是 batch_size
h0 = cell.zero_state(32, np.float32) # 通过zero_state得到一个全0的初始状态，形状为(batch_size, state_size)
output, h1 = cell.call(inputs, h0) #调用call函数

print(h1.shape) # (32, 128)


'''
#对于BasicLSTMCell，情况有些许不同，因为LSTM可以看做有两个隐状态h和c，对应的隐层就是一个Tuple，每个都是(batch_size, state_size)的形状

'''
lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=128)
inputs = tf.placeholder(np.float32, shape=(32, 100)) # 32 是 batch_size
h0 = lstm_cell.zero_state(32, np.float32) # 通过zero_state得到一个全0的初始状态
output, h1 = lstm_cell.call(inputs, h0)

print(h1.h)  # shape=(32, 128)
print(h1.c)  # shape=(32, 128)



#########################################
#   tf.nn.dynamic_rnn
#########################################

# inputs: shape = (batch_size, time_steps, input_size) 
# cell: RNNCell
# initial_state: shape = (batch_size, cell.state_size)。初始状态。一般可以取零矩阵
outputs, state = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state)



#########################################
#   堆叠RNNCell：MultiRNNCell
#########################################


# 每调用一次这个函数就返回一个BasicRNNCell
def get_a_cell():
    return tf.nn.rnn_cell.BasicRNNCell(num_units=128)
# 用tf.nn.rnn_cell MultiRNNCell创建3层RNN
cell = tf.nn.rnn_cell.MultiRNNCell([get_a_cell() for _ in range(3)]) # 3层RNN
# 得到的cell实际也是RNNCell的子类
# 它的state_size是(128, 128, 128)
# (128, 128, 128)并不是128x128x128的意思
# 而是表示共有3个隐层状态，每个隐层状态的大小为128
print(cell.state_size) # (128, 128, 128)
# 使用对应的call函数
inputs = tf.placeholder(np.float32, shape=(32, 100)) # 32 是 batch_size
h0 = cell.zero_state(32, np.float32) # 通过zero_state得到一个全0的初始状态
output, h1 = cell.call(inputs, h0)
print(h1) # tuple中含有3个32x128的向量