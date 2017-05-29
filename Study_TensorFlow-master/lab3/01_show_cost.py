#-*- coding: utf-8 -*-

from __future__ import print_function

import tensorflow as tf
from matplotlib import pyplot as plt

# Graph Input
X = [1., 2., 3.]
Y = [1., 2., 3.]
m = n_smaples = len(X)   # m은 X리스트의 길이이다. (여기서는 3이 된다)
# n_smaples 는 왜있는거지 쓰지도 않으니까 지워도 될 듯


W = tf.placeholder(tf.float32)

hypothesis = tf.multiply(X, W)    # hypothesis = W * X

#pow(,2) 로 2승을 계산,    reduce_sum 으로 시그마를 취한 후 m 으로 나눠서 평균 계산
cost = tf.reduce_sum(tf.pow(hypothesis - Y, 2)) / m

init = tf.global_variables_initializer()

#그래프를 표시하기 위해 데이터를 누적할 리스트, append 함수로 누적시킬 수 있다.
W_val = []
cost_val = []

sess = tf.Session()
sess.run(init)

# 0.1 단위로 증가할 수 없어서 -30부터 시작. 그래프에는 -3에서 5까지 표시됨
for i in range(-30, 51):
    xPos = i*0.1
    yPos = sess.run(cost, feed_dict={W : xPos}) # cost 함수의 변화를 확인하기 위해 cost함수를 실행
    print('{:9.7f}, {:9.7f}'.format(xPos,yPos)) # 존나 보기좋게 나온다

    W_val.append(xPos) # 루프가 돌때마다 xPos 값을 W_val 에 누적시킴(append 함수)
    cost_val.append(yPos) # 위와 내용이 같습니당


# for i in range(-30, 50):    #append >> 값 추가
#     print(i * -0.1, sess.run(cost, feed_dict={W: i * 0.1}))
#     W_val.append(i * 0.1)
#     cost_val.append(sess.run(cost, feed_dict={W: i * 0.1}))



#그래프를 실행시키는 코드
plt.plot(W_val, cost_val, 'ro') #주어진 데이터들을 점으로 표시
plt.ylabel('cost')#y축 라벨정의
plt.xlabel('W')  # x축 라벨정의
plt.show() # 그래프를 표시
