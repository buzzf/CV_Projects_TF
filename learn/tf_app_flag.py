#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Author: zzf
# @Date:   2018-08-28 20:44:36
# @Last Modified by:   zzf
# @Last Modified time: 2018-08-28 20:45:52

import tensorflow  as tf


FLAGS=tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('flag_float', 0.01, 'input a float')
tf.app.flags.DEFINE_integer('flag_int', 400, 'input a int')
tf.app.flags.DEFINE_boolean('flag_bool', True, 'input a bool')
tf.app.flags.DEFINE_string('flag_string', 'yes', 'input a string')
 
print(FLAGS.flag_float)
print(FLAGS.flag_int)
print(FLAGS.flag_bool)
print(FLAGS.flag_string)
