#-*- coding: utf-8 -*-

# from __future__ import print_function
#
# import numpy as np #  loadtxt를 사용하기 위해참조
# import tensorflow as tf
#
# xy = np.loadtxt('train.txt', unpack=True, dtype='float32') # train.txt  >> 파일이름 , delimiter >> 데이터를 나누는기준 ,  dtype >> 파일의 자료형
# x_data = xy[0:-1] #x_data 의 첫번째 원소에서 그 원소의 마지막 값들 만 가져옴
# y_data = xy[-1] #y_data 의 모든 원소에서 그 원소의 마지막 값들 만 가져옴
#
# W = tf.Variable(tf.random_uniform([1, 3], -1.0, 1.0))
#
# hypothesis = tf.matmul(W, x_data)
#
# cost = tf.reduce_mean(tf.square(hypothesis - y_data))
#
# learning_rate = 0.1
# optimizer = tf.train.GradientDescentOptimizer(learning_rate)
# train = optimizer.minimize(cost)
#
# init = tf.global_variables_initializer()
#
# sess = tf.Session()
# sess.run(init)
#
# for step in range(2001):
#     sess.run(train)
#     if step % 20 == 0:
#         print(step, sess.run(cost), sess.run(W))

from __future__ import print_function

import numpy as np
import tensorflow as tf

xy = np.loadtxt('train.txt', unpack=True, dtype='float32')
x_data = xy[0:-1]
y_data = xy[-1]

W = tf.Variable(tf.random_uniform([1,3],-1.0, 1.0))
hypothesis = tf.matmul(W, x_data)

cost = tf.reduce_mean(tf.square(hypothesis - y_data))

learning_rate=0.1
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(cost)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for step in xrange(2001):
    sess.run(train)
    if step%20==0:
        print (step, sess.run(cost), sess.run(W))