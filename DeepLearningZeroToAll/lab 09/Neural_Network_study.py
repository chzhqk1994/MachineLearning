import tensorflow as tf
import numpy as np

# # x1 x2 y
#   0  0  0
#   0  1  1
#   1  0  1
#   1  1  0
xy = np.loadtxt('07train.txt', unpack=True) # 열 단위로 읽어들인다

x_data = np.transpose(xy[:-1])
# [[0,0],
#  [0,1],
#  [1,0],
#  [1,1]]

y_data = np.reshape(xy[-1], (4, 1))
# [[0],
#  [1],
#  [1],
#  [0]]

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_uniform([2, 2], -1.0, 1.0))
W2 = tf.Variable(tf.random_uniform([2, 1], -1.0, 1.0))

b1 = tf.Variable(tf.zeros([2]))  # [0., 0.] b1에 해당하는 Weight(여기선 W1)의 모양이 [a, b]이면 b 를 넣으면 된다.
b2 = tf.Variable(tf.zeros([1]))  # [0.]

L2 = tf.sigmoid(tf.matmul(X, W1) + b1)  # 히든레이어갯수는 제한이 없지만 너무 많아지면 overfitting 문제가 발생할 수 있다
hypothesis = tf.sigmoid(tf.matmul(L2, W2) + b2)  # 앞에서 연산된 L2를 sigmoid 함수에 넣고 실행

cost = -tf.reduce_mean(Y*tf.log(hypothesis) + (1-Y)*tf.log(1-hypothesis))

learning_rate = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print('b1: ', sess.run(b1))
    print('b2: ', sess.run(b2))
    for step in range(10000):
        sess.run(train, feed_dict={X: x_data, Y: y_data})  # 학습시키는 코드

        if step%1000 == 999:
            r_cost, (r_W1, r_W2) = sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run([W1, W2])  # 보기 좋게 바꾸기위해(reshape) 하기 위해 변수에 넣음
            print('{:5} {:10.8f} {} {}'.format(step + 1, r_cost, np.reshape(r_W1, (1, 4)), np.reshape(r_W2, (1, 2))))
    print('------------------------------------------------------')

    correct_prediction = tf.equal(tf.floor(hypothesis + 0.5), Y)

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
    param = [hypothesis, tf.floor(hypothesis + 0.5), correct_prediction, accuracy] # tf.floor(hypothesis + 0.5) >> 0.5 를 더하고 소수점 이하는 버린다
    result = sess.run(param, feed_dict={X: x_data, Y: y_data})

    print(*result[0])  # hypothesis
    print(*result[1])  # tf.floor( hypothesis + 0.5)
    print(*result[2])  # correct_prediction
    print( result[-1])  # accuracy
    print('Accuracy : ', accuracy.eval({X: x_data, Y: y_data}))

    print('b1: ', sess.run(b1))
    print('b2: ', sess.run(b2))
