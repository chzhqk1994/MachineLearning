#-*- coding: utf-8 -*-

from __future__ import print_function


import tensorflow as tf
from matplotlib import pyplot as plt

# Graph Input
X = [1., 2., 3.]
Y = [1., 2., 3.]
m = n_smaples = len(X)

# model weight
W = tf.placeholder(tf.float32)

# Construct a linear model
hypothesis = tf.multiply(X, W)

# Cost function
cost = tf.reduce_sum(tf.pow(hypothesis - Y, 2)) / m

init = tf.global_variables_initializer()

# for graphs
W_val = []
cost_val = []

# Launch the graphs
sess = tf.Session()
sess.run(init)

for i in xrange(-30, 50):    #append >> 값 추가
    print(i * -0.1, sess.run(cost, feed_dict={W: i * 0.1}))
    W_val.append(i * 0.1)
    cost_val.append(sess.run(cost, feed_dict={W: i * 0.1}))

plt.plot(W_val, cost_val, 'ro') #주어진 데이터들을 점으로 표시
plt.ylabel('cost')#y축 라벨정의
plt.xlabel('W')  # x축 라벨정의
plt.show()