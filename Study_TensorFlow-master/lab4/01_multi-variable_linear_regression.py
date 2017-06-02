from __future__ import print_function

import tensorflow as tf

x_data = [[1., 0., 3., 0., 5.],
            [0., 2., 0., 4., 0.]]
y_data = [2, 4, 6, 8, 10]

W = tf.Variable(tf.random_uniform([1,2], -1.0, 1.0)) # x_data와 곱하므로 [ , ]행렬 크기를 맞춰준다
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

hypothesis = tf.matmul(W,x_data) + b
cost = tf.reduce_mean(tf.square(hypothesis - y_data))

learning_rate = 0.1
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(cost)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for step in range(2001):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(cost), sess.run(W), sess.run(b))

# from __future__ import print_function
#
# import tensorflow as tf
#
# class Multi_Variable_Linear_Regression:
#     def __init__(self):
#         self.x1_data = [1, 0, 3, 0, 5]
#         self.x2_data = [0, 2, 0, 4, 0]
#         self.y_data = [1,2,3,4,5]
#
#         self.W1 = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
#         self.W2 = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
#
#         self.b= tf.Variable(tf.random_uniform([1], -1.0, 1.0))
#
#         self.hypothesis = self.W1 * self.x1_data + self.W2 * self.x2_data + self.b
#
#         self.cost = tf.reduce_mean(tf.square(self.hypothesis - self.y_data))
#
#     def gradient_descent(self, a):
#         self.learning_rate = a
#         self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
#         self.train = self.optimizer.minimize(self.cost)
#
#     def initialize(self):
#         self.init = tf.global_variables_initializer()
#         self.sess=tf.Session()
#         self.sess.run(self.init)
#
#
# mvl=Multi_Variable_Linear_Regression()
# mvl.gradient_descent(0.01)
# mvl.initialize()
# for step in range(2001):
#     mvl.sess.run(mvl.train)
#     if step%20==0:
#         print (step, mvl.sess.run(mvl.cost), mvl.sess.run(mvl.W1), mvl.sess.run(mvl.W2), mvl.sess.run(mvl.b))