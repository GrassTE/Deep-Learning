import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.utils import shuffle
from sklearn.preprocessing import scale
from Script.Boston.model import *



df = pd.read_csv("boston.csv", header=0)
ds = df.values
x_data = ds[:, :12]
y_data = ds[:, 12]

for i in range(10):
    x_data[:, i]=(x_data[:,i]-x_data[:,i].min())/(x_data[:,i].max()-x_data[:,i].min())

train_num = 300
valid_num = 100
test_num = len(x_data) - train_num - valid_num
#划分训练集
x_train = x_data[:train_num]
y_train = y_data[:train_num]
#划分验证集
x_valid = x_data[train_num : train_num + valid_num]
y_valid = y_data[train_num : train_num + valid_num]
#划分测试集
x_test = x_data[train_num + valid_num : train_num + valid_num + test_num]
y_test = y_data[train_num + valid_num : train_num + valid_num + test_num]

#数据类型转换
x_train = tf.cast(scale(x_train), dtype=tf.float32)
x_valid = tf.cast(scale(x_valid), dtype=tf.float32)
x_test = tf.cast(scale(x_test), dtype=tf.float32)
#创建变量
W = tf.Variable(tf.random.normal([12, 1],mean=0.0, stddev=1.0, dtype=tf.float32))
B = tf.Variable(tf.zeros(1), dtype=tf.float32)

training_epochs = 50 #迭代次数
learning_rate = 0.001 # 学习率
batch_size = 10 # 批量训练一次的样本数

optimizer = tf.keras.optimizers.SGD(learning_rate)

loss_list_train = []
loss_list_valid = []
total_step = int(train_num/batch_size)

for epoch in range(training_epochs):
    for step in range(total_step):
        xs = x_train[step * batch_size:(step+1)*batch_size, :]
        ys = y_train[step * batch_size:(step+1)*batch_size]
        grads = grad(xs, ys, W, B)
        optimizer.apply_gradients(zip(grads, [W, B]))

    loss_train = loss(x_train, y_train, W, B).numpy()
    loss_valid = loss(x_valid, y_valid, W, B).numpy()
    loss_list_train.append(loss_train)
    loss_list_valid.append(loss_valid)
    print("epoch={:3d} ,train_loss={:.4f},valid_loss={:.4f}".format(epoch+1,loss_train,loss_valid))

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.plot(loss_list_train,'blue',label='Train Loss')
plt.plot(loss_list_valid,'red',label='Valid Loss')
plt.legend(loc=1)
plt.show()





