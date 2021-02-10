import tensorflow as tf
tf.compat.v1.disable_eager_execution()

value = tf.Variable(0, name="value")
one = tf.constant(1)
new_value = tf.add(value, one)
update_value = tf.compat.v1.assign(value, new_value)

init = tf.compat.v1.global_variables_initializer()

with tf.compat.v1.Session() as sess:
    sess.run(init)
    for _ in range(10):
        sess.run(update_value)
        print(sess.run(value))