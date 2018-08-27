# 变量
import tensorflow as tf

# 定义一个变量
x = tf.Variable([1, 2])

a = tf.constant([3, 3])

# 变量为初始化不能使用, 使用该段代码可以对全局变量进行初始化
init = tf.global_variables_initializer()

# 添加一个减法op
sub = tf.subtract(x, a)

# 添加一个加法op
add = tf.add(x, sub)

with tf.Session() as sess:
    # 执行全局变量初始化
    sess.run(init)
    print(sess.run(sub))
    print(sess.run(add))


# 累加
# 创建一个变量初始化为0
state = tf.Variable(0, name='counter')
# 穿件一个op，作用是使state加1
new_value = tf.add(state, 1)

# 赋值操作，将new_value的值赋给state
update = tf.assign(state, new_value)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(state))
    for _ in range(5):
        sess.run(update)
        print(sess.run(state))
