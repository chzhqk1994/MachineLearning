from __future__ import print_function

import numpy as np
import tensorflow as tf

xy = np.loadtxt('train.txt', unpack=True, dtype='float32')
x_data = np.transpose(xy[0:3])
y_data = np.transpose(xy[3:])

X = tf.placeholder("float", [None, 3])
Y = tf.placeholder("float", [None, 3])

W = tf.Variable(tf.zeros([3, 3]))

# matrix shape X=[8, 3], W=[3, 3]
hypothesis = tf.nn.softmax(tf.matmul(X, W))  # X*W 를 softmax 연산처리하여 hypothesis에 넣음

learning_rate = 0.001

cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), reduction_indices=1)) # softmax처리된 hypothesis를 cross-entropy cost 에 넣고 실행. 근데 reduction 이건 왜하는거지
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)  # GradientDescent 알고리즘 실행

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for step in range(2001):
        sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
        if step % 200 == 0:
            print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W))

    print('--------------------')

    a = sess.run(hypothesis, feed_dict={X: [[1, 11, 7]]})
    print(a, sess.run(tf.argmax(a, 1)))

    b = sess.run(hypothesis, feed_dict={X: [[1, 3, 4]]})
    print(a, sess.run(tf.argmax(b, 1)))

    c = sess.run(hypothesis, feed_dict={X: [[1, 1, 0]]})
    print(a, sess.run(tf.argmax(c, 1)))

    all = sess.run(hypothesis, feed_dict={X: [[1, 11, 7], [1, 3, 4], [1, 1, 0]]})
    print(all, sess.run(tf.argmax(all, 1)))
