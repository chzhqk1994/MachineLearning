#-*- coding: utf-8 -*-

import tensorflow as tf

sess=tf.Session()

a= tf.constant(2)
b= tf.constant(3)

c=a+b


print(a)
print(b)
print(c)  #세션 실행을 하지 않으면 텐서의 정보만 출력

print (sess.run(a))
print (sess.run(b))
print (sess.run(c))
print (sess.run(a+b)) #우리가 원하는 텐서 값이 출력