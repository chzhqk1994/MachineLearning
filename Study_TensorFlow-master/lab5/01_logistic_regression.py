from __future__ import print_function

import numpy as np
import tensorflow as tf

class Logistic_Regression:
    def __init__(self):
        self.xy = np.loadtxt('train.txt', unpack=True, dtype='float32')
        self.x_data = self.xy[0:-1]
        self.y_data = self.xy[-1]

        self.X = tf.placeholder(tf.float32)
        self.Y = tf.placeholder(tf.float32)

        self.W = tf.Variable(tf.random_uniform([1, len(self.x_data)], -1.0, 1.0))
        self.h = tf.matmul(self.W, self.X)
    def Hypothesis(self):
        self.hypothesis = tf.div(1., 1. + tf.exp(-self.h))
        self.cost = -tf.reduce_mean(self.Y * tf.log(self.hypothesis) + (1 - self.Y) * tf.log(1 - self.hypothesis))

    def gradient_descent(self, a):
        self.learning_rate = a
        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        self.train = self.optimizer.minimize(self.cost)

    def initialize(self):
        self.init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(self.init)


lor=Logistic_Regression()
lor.Hypothesis()
lor.gradient_descent(0.1)
lor.initialize()
for step in range(2001):
    lor.sess.run(lor.train, feed_dict={lor.X: lor.x_data, lor.Y: lor.y_data})

    if step % 20 == 0:
        print(step, lor.sess.run(lor.cost, feed_dict={lor.X: lor.x_data, lor.Y: lor.y_data}), lor.sess.run(lor.W))
