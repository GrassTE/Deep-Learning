import tensorflow as tf

tf.compat.v1.disable_eager_execution()

node1 = tf.constant(3.0,tf.float32,name="node1")
node2 = tf.constant(4.0,tf.float32,name="node2")
node3 = tf.add(node1,node2)

print(node3)
print(node1)
print(node2)

sess = tf.compat.v1.Session()
print("node1的结果：",sess.run(node1))
print("node1的结果：",sess.run(node3))

sess.close()