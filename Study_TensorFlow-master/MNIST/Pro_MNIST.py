from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# 레이어 중첩으로 만들어 볼 예정

mnist = input_data.read_data_sets('./samples/MNIST_data', one_hot=True)

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))



# sess.run(global_variables.initializer())

learning_rate = 0.01
activation = tf.nn.softmax(tf.matmul(X,W) + b)
cost = -tf.reduce_sum(Y * tf.log(activation))
train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(1000):
    batch_x, batch_y = mnist.train.next_batch(100)
    sess.run(train, feed_dict={X: batch_x, Y:batch_y})

correct_prediction = tf.equal(tf.argmax(activation, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))