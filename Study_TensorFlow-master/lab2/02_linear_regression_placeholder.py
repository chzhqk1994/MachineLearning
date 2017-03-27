#-*- coding: utf-8 -*-

# from __future__ import print_function
#
# import tensorflow as tf
#
# x_data = [1, 2, 3]
# y_data = [1, 2, 3]
#
# W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
# b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
#
# X = tf.placeholder(tf.float32)
# Y = tf.placeholder(tf.float32)
#
# hypothesis = W * X + b
#
# cost = tf.reduce_mean(tf.square(hypothesis - Y))
#
# a = tf.Variable(0.1)
# optimizer = tf.train.GradientDescentOptimizer(a)
# train = optimizer.minimize(cost)
#
# init = tf.global_variables_initializer()
#
# sess = tf.Session()
# sess.run(init)
#
# for step in range(2001):
#     sess.run(train, feed_dict={X: x_data, Y: y_data})
#     if step % 20 == 0:
#         print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W), sess.run(b))
#
# print(sess.run(hypothesis, feed_dict={X: 5}))
# print(sess.run(hypothesis, feed_dict={X: 2.5}))

import tensorflow as tf

class Linear_regression:
    def __init__(self):
        self.x_data = [1, 2, 3]
        self.y_data = [1, 2, 3]

        self.W = tf.Variable(tf.random_uniform([1],-1.0, 1.0))
        self.b = tf.Variable(tf.random_uniform([1],-1.0, 1.0))

        self.X = tf.placeholder(tf.float32)
        self.Y = tf.placeholder(tf.float32)


    def Hypothesis(self):
        self.hypothesis = self.W * self.X + self.b
        self.cost = tf.reduce_mean(tf.square(self.hypothesis - self.Y))


    def gradient_descent(self,Learning_rate):   #Learning rate 을 입력받 을 수있음
        self.a = tf.Variable(Learning_rate)
        self.optimizer = tf.train.GradientDescentOptimizer(self.a)
        self.train = self.optimizer.minimize(self.cost)


    def initialize(self):
        self.init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(self.init)


lr = Linear_regression()
lr.Hypothesis()
lr.gradient_descent(0.1)
lr.initialize()
for step in xrange(2001):
    lr.sess.run(lr.train, feed_dict={lr.X: lr.x_data, lr.Y : lr.y_data})
    if step%20==0:
        print(step, lr.sess.run(lr.cost, feed_dict={lr.X: lr.x_data, lr.Y: lr.y_data}), lr.sess.run(lr.W), lr.sess.run(lr.b))