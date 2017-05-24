#-*- coding: utf-8 -*-

import tensorflow as tf

a=tf.placeholder(tf.int16)
b=tf.placeholder(tf.int16)

add=tf.add(a,b)
mul=tf.multiply(a,b)

print (add)
print (a+b)
print (mul)
print (a*b)

with tf.Session() as sess:
    print (sess.run(add, feed_dict={a:2, b:3})) #add 텐서서 사용하는 placeholder a 와 b에 각각 2와 3을 넣는다  a와 b의 순서가 바뀌어도 무관

    print (sess.run(mul, feed_dict={a:5, b:4}))

    feed = {a: 8, b: 5}
    print (sess.run(mul, feed))