import tensorflow as tf
tf.compat.v1.disable_eager_execution()

#定义计算图
tens1 = tf.constant([1,2,3])

#创建会话
sess = tf.compat.v1.Session()

#捕捉异常
try:
    print(tens1)
    print(sess.run(tens1))
except:
    print("Exception!")
finally:
    sess.close()

