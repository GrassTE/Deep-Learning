import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def model(x, w, b):
    return tf.multiply(x, w) + b

if __name__ == '__main__':
    tf.compat.v1.disable_eager_execution()

    # 设置随机数种子
    #np.random.seed(5)

    x_data = np.linspace(-1, 1, 100)
    y_data = 2 * x_data + 1.0 + np.random.randn(*x_data.shape) * 0.4

    x = tf.compat.v1.placeholder("float", name='x')
    y = tf.compat.v1.placeholder("float", name='y')

    w = tf.Variable(1.0, name='w0')
    b = tf.Variable(0.0, name='b0')
    pred = model(x, w, b)
    train_epochs = 10
    learning_rate = 0.05

    loss_function = tf.reduce_mean(tf.square(y-pred))
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(loss_function)
    sess = tf.compat.v1.Session()
    init = tf.compat.v1.global_variables_initializer()
    sess.run(init)

    for epoch in range(train_epochs):
        for xs,ys in zip(x_data, y_data):
            _, loss = sess.run([optimizer,loss_function], feed_dict={x:xs, y:ys})
        b0temp = b.eval(session=sess)
        w0temp = w.eval(session=sess)
        plt.plot(x_data, w0temp * x_data + b0temp)


    plt.scatter(x_data, y_data)
    #plt.plot(x_data, 1.0 + 2 * x_data, color='red', linewidth=3)
    print("w:", sess.run(w))
    print("b:", sess.run(b))
    plt.show()


