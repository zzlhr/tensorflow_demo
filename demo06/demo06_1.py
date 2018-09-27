# MNIST数据集分类简单版本
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 载入数据集  one_hot 转换 为0和1的视图
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 每个批次的大小
batch_size = 50

# 计算一共有多少个批次
n_batch = mnist.train.num_examples // batch_size




# 定义两个placeholder
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])


# 创建一个简单的神经网络
W = tf.Variable(tf.zeros([784, 10]), name='W')
b = tf.Variable(tf.zeros([10]),name='b')
prediction = tf.nn.softmax(tf.matmul(x, W) + b, name='prediction')

# 二次代价函数
loss = tf.reduce_mean(tf.square(y - prediction))
# 使用梯度下降法
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# 初始化变量
init = tf.global_variables_initializer()

# 结果
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))

# 求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(50):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            print(batch_xs)
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})

        # 测试准确率
        acc = sess.run(accuracy, 
        feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print("Item" + str(epoch) + ", Testing Accuracy " + str(acc))
    
    saver.save(sess, '/Users/lhr/tensorflow_demo/demo06/model/model_save')
    