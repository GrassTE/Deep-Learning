import tensorflow as tf
tf.compat.v1.disable_eager_execution()

#定义计算图
tens1 = tf.constant([1,2,3])

#创建会话
sess = tf.compat.v1.Session()

#上下文管理器
with tf.compat.v1.Session() as sess:
    print(tens1)
    print(sess.run(tens1))

