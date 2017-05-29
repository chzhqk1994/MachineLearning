#-*- coding: utf-8 -*-

from __future__ import print_function
import tensorflow as tf

x_data=[1, 2, 3] # x와 y가 1:1 이므로 최적의 학습은 W : 1, b : 0 이다.(아래의 가설을 기준)
y_data=[1, 2, 3]

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0)) # -1.0과 1.0 사이의 정규분포 값 1개를 생성한다.
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

hypothesis = W * x_data + b # x_data 가 3개이므로 이 연산은 3번 반복된다. 물론 b 또한 3번 더해진다

cost = tf.reduce_mean(tf.square(hypothesis - y_data))

a=tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

init = tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)

for step in range(2001): # 0부터 시작하며 2001은 포함되지 않는다.  즉 0부터 2000 까지 반복된다.
    sess.run(train) #연산은 여기서 끝나지만 W와 b의 값을 찾아가는 과정을 시각적으로 나타내기 위해 아래의 코드를 작성한다
    if step%20==0:
        print (step, sess.run(cost), sess.run(W), sess.run(b))
        # print('{:4} {} {} {}'.format(step, sess.run(cost), sess.run(W), sess.run(b)))       # 똑같이 출력되는데 뭔가 다르다 소숫점이 쥰내 길게 나옴