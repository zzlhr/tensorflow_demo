import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image
import numpy as np


im = Image.open('/Users/lhr/im.jpeg')
out = im.resize((28, 28))

out.save('/Users/lhr/im1.jpeg')
im = Image.open('/Users/lhr/im1.jpeg')

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
W = tf.Variable(tf.zeros([784, 10]), name='W')
b = tf.Variable(tf.zeros([10]), name='b')
prediction = tf.nn.softmax(tf.matmul(x, W) + b)
result = tf.argmax(prediction, 1)

init = tf.global_variables_initializer()
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init)
    saver.restore(sess, '/Users/lhr/tensorflow_demo/demo06/model/model_save')
    im = im.convert("L")
    im.show()
    data = im.getdata()
    data = np.matrix(data, dtype='float32')
    x_data = tf.reshape(data,[-1, 784])
    print(x_data)
    y_conv2 = sess.graph.get_tensor_by_name("prediction:0")
    sess.run(y_conv2, feed_dict={x: x_data})
