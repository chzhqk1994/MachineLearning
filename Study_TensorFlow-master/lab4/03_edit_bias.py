from __future__ import print_function

import tensorflow as tf

# x_data = [[1., 0., 3., 0., 5.],
#           [0., 2., 0., 4., 0.]]
#
# y_data = [1,2,3,4,5]
#
# W = tf.Variable(tf.random_uniform([1,2], -1.0, 1.0))
# b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))               #b를 따로 만든 상태
#
# hypothesis = tf.matmul(W,x_data) + b

# ========================================================위의 코드와 비교========================================

x_data = [[1., 1., 1., 1., 1.],  # [1,1,1,1,1] >> b(bias) 를 행렬에 넣은 값 이다.
          [1., 0., 3., 0., 5.],
          [0., 2., 0., 4., 0.]]  # 3X1 행렬  >> W는 1X3 행렬이 되어야 한다
y_data = [3, 5, 7, 9, 11]

W = tf.Variable(tf.random_uniform([1, 3], -1.0, 1.0))  # 이전 코드에서 b를 삭제하고 행렬의 크기를 1X3 으로 늘림

hypothesis = tf.matmul(W, x_data) # 행렬 W와 x_data 를 곱한다


cost = tf.reduce_mean(tf.square(hypothesis - y_data))

learning_rate = 0.1
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(cost)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in range(2001):
    sess.run(train)
    if step%20==0:
        print (step, sess.run(cost), sess.run(W)) # b가 배열속으로 들어갔으므로 b가 사라짐


# from __future__ import print_function
#
# import tensorflow as tf
#
# x_data = [[1., 1., 1., 1., 1.],
#           [1., 0., 3., 0., 5.],
#           [0., 2., 0., 4., 0.]]
#
# y_data = [1, 2, 3, 4, 5]
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
