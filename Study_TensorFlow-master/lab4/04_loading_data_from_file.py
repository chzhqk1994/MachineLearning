# -*- coding: utf-8 -*-

from __future__ import print_function

import numpy as np #  loadtxt를 사용하기 위해참조, numpy는 파이썬에서 과학적 계산을 위한 핵심 라이브러리임. 이것은 고성능 다차원 배열 객체와 이들 배열과 함께 작동하는 도구들을 제공함.
import tensorflow as tf

xy = np.loadtxt('train.txt', unpack=True, dtype='float32') # train.txt  >> 파일이름 , delimiter >> 데이터를 나누는기준 ,  dtype >> 파일의 자료형
                                                                # #으로 시작하는 줄은 주석으로 판단하고 읽지 않는다.
                                                                # unpack 속성은 디폴트 값은 그대로 읽어오고, true를 전달하면 전치시켜서 읽어들인다

x_data = xy[0:-1] #x_data 의 첫번째 원소에서 그 원소의 마지막 값들 만 가져옴
y_data = xy[-1] #y_data 의 모든 원소에서 그 원소의 마지막 값들 만 가져옴

W = tf.Variable(tf.random_uniform([1, len(x_data)], -1.0, 1.0))  # W는 1x (x_data의 크기) 형태의 행렬

hypothesis = tf.matmul(W, x_data)

cost = tf.reduce_mean(tf.square(hypothesis - y_data))

learning_rate = 0.1
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(cost)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

print(x_data)
print('\n')
print(y_data)
print('\n')

for step in range(2001):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(cost), sess.run(W))