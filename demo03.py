import tensorflow as tf

# Fetch

input1 = tf.constant(3.0)
input2 = tf.constant(2.0)
input3 = tf.constant(5.0)

add = tf.add(input2, input3)
mul = tf.multiply(input1, add)

with tf.Session() as sess:
    # 同时运行多个op
    result = sess.run([mul, add])
    print(result)


# Feed
# 创建占位符
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1, input2)


with tf.Session() as sess:
    print(sess.run())