#-*- coding: utf-8 -*-

from __future__ import print_function

import tensorflow as tf

x_data = [1, 2, 3]
y_data = [3, 6, 9]

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

hypothesis = W * X + b

cost = tf.reduce_mean(tf.square(hypothesis - Y))

a = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for step in range(2001):
    sess.run(train, feed_dict={X: x_data, Y: y_data}) # train의 X에 x_data 를 넣고 Y에 y_data 를 넣은 후 연산 실행
    if step % 20 == 0:
        print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W), sess.run(b)) # cost 값의 변화를 보기위해 출력    >> 근데 feed_dict 를 왜 하는건지 모르겠다 train 에서 연산이 끝난게 아닌건가
                                                                                                    # train 으로 학습은 끝냈지만 cost 값 변화를 확인하기 위해 cost 만 따로 연산 시킨 듯

print(sess.run(hypothesis, feed_dict={X: 5})) #train 을 실행하여 학습이 끝내고 X가 5일 때의 Y값을 물어본다
print(sess.run(hypothesis, feed_dict={X: 2.5}))
# 학습시킬땐 train 을 run 하고 예측을 할 땐 hypothesis에 feed_dict를 하고 run 시킨다.





# import tensorflow as tf
#
# class Linear_regression:
#     def __init__(self):
#         self.x_data = [1, 2, 3]
#         self.y_data = [1, 2, 3]
#
#         self.W = tf.Variable(tf.random_uniform([1],-1.0, 1.0))
#         self.b = tf.Variable(tf.random_uniform([1],-1.0, 1.0))
#
#         self.X = tf.placeholder(tf.float32)
#         self.Y = tf.placeholder(tf.float32)
#
#
#     def Hypothesis(self):
#         self.hypothesis = self.W * self.X + self.b
#         self.cost = tf.reduce_mean(tf.square(self.hypothesis - self.Y))
#
#
#     def gradient_descent(self,Learning_rate):   #Learning rate 을 입력받 을 수있음
#         self.a = tf.Variable(Learning_rate)
#         self.optimizer = tf.train.GradientDescentOptimizer(self.a)
#         self.train = self.optimizer.minimize(self.cost)
#
#
#     def initialize(self):
#         self.init = tf.global_variables_initializer()
#         self.sess = tf.Session()
#         self.sess.run(self.init)
#
#
# lr = Linear_regression()
# lr.Hypothesis()
# lr.gradient_descent(0.1)
# lr.initialize()
# for step in range(2001):
#     lr.sess.run(lr.train, feed_dict={lr.X: lr.x_data, lr.Y : lr.y_data})
#     if step%20==0:
#         print(step, lr.sess.run(lr.cost, feed_dict={lr.X: lr.x_data, lr.Y: lr.y_data}), lr.sess.run(lr.W), lr.sess.run(lr.b))