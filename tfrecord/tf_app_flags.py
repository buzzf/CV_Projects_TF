import tensorflow  as tf  


# tf.app.flags.DEFINE_xxx()就是添加命令行的optional argument（可选参数），而tf.app.flags.FLAGS可以从对应的命令行参数取出参数。

FLAGS=tf.app.flags.FLAGS  
tf.app.flags.DEFINE_float(  
    'flag_float', 0.01, 'input a float')  
tf.app.flags.DEFINE_integer(  
    'flag_int', 400, 'input a int')  
tf.app.flags.DEFINE_boolean(  
    'flag_bool', True, 'input a bool')  
tf.app.flags.DEFINE_string(  
    'flag_string', 'yes', 'input a string')  
  
print(FLAGS.flag_float)  
print(FLAGS.flag_int)  
print(FLAGS.flag_bool)  
print(FLAGS.flag_string) 