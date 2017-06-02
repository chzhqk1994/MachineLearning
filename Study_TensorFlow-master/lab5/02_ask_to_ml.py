# Logistic Classification

from __future__ import print_function

import numpy as np
import tensorflow as tf

# #x0 x1 x2 y
#  1  2  1  0
#  1  3  2  0
#  1  3  4  0
#  1  5  5  1
#  1  7  5  1
#  1  2  5  1
xy = np.loadtxt('train.txt', unpack=True, dtype='float32')  # unpack 이 True 이므로 전치행렬도 받아온다.
x_data = xy[0:-1]  # 0번째부터 마지막 직전까지
y_data = xy[-1]  # 마지막 것 만

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_uniform([1, len(x_data)], -1.0, 1.0))

h = tf.matmul(W, X)
hypothesis = tf.div(1., 1. + tf.exp(-h))  #  H(x) = 1/(1+e^-WX)

cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))

learning_rate = 0.1
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(cost)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in range(2001):
    sess.run(train, feed_dict={X: x_data, Y: y_data})

    if step % 20 == 0:
        print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W))


print('---------------------------')

print("[[1], [2], [2]] : ", sess.run(hypothesis, feed_dict={X: [[1], [2], [2]]}))
print("[[1], [5], [5]] : ", sess.run(hypothesis, feed_dict={X: [[1], [5], [5]]}))
print("[[1, 1], [4, 3], [3 ,5]] : ", sess.run(hypothesis, feed_dict={X: [[1, 1], [4, 3], [3, 5]]}))

print("[1, 4, 2] [1 ,0, 10] : ", end=' ')
print(sess.run(hypothesis, feed_dict={X: [[1, 1], [4, 0], [2, 10]]}))
# [[1,1],
#  [4,0],
#  [2,10]]   >> [1,4,2], [1,0,10] 이렇게 들어간다
