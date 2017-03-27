#-*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np
import tensorflow as tf



class Softmax_classification:
    def __init__(self):

        self.xy = np.loadtxt('train.txt', unpack=True, dtype='float32')
        self.x_data = np.transpose(self.xy[0:3])  #transpose : 전치행렬
        self.y_data = np.transpose(self.xy[3:])


        self.X = tf.placeholder("float", [None, 3])
        self.Y = tf.placeholder("float", [None, 3])

        self.W = tf.Variable(tf.zeros([3, 3]))

    def Hypothesis(self):
        # matrix shape X=[8, 3], W=[3, 3]
        self.hypothesis = tf.nn.softmax(tf.matmul(self.X, self.W))

        self.learning_rate = 0.001

    def gradient_descent(self):
        self.cost = tf.reduce_mean(-tf.reduce_sum(self.Y * tf.log(self.hypothesis), reduction_indices=1))
        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost)



    def initialize(self):
        self.init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(self.init)


sc=Softmax_classification()
sc.Hypothesis()
sc.gradient_descent()
sc.initialize()


with tf.Session() as sess:
    sc.sess.run(sc.init)

    for step in range(2001):
        sc.sess.run(sc.optimizer, feed_dict={sc.X: sc.x_data, sc.Y: sc.y_data})
        if step % 20 == 0:
            print(step, sc.sess.run(sc.cost, feed_dict={sc.X: sc.x_data, sc.Y: sc.y_data}), sc.sess.run(sc.W))
