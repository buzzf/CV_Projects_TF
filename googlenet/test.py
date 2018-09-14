import tensorflow as tf

label_lookup_path = 'inception_model/imagenet_2012_challenge_label_map_proto.pbtxt'

proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()

print(len(proto_as_ascii))
print(proto_as_ascii[0])
print(proto_as_ascii[1])
print(proto_as_ascii[2])
print(proto_as_ascii[3])
print(proto_as_ascii[4])