# NeuralNetworkForMnist.py : 94.57%

import tensorflow as tf

# Import MINST data
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Parameters. 반복문에서 사용하는데, 미리 만들어 놓았다.
learning_rate = 0.001
training_epochs = 15
batch_size = 100
display_step = 1

# tf Graph Input
X = tf.placeholder(tf.float32, [None, 784])  # mnist dataset은 그림 하나당 28*28의 사이즈를 갖고 있다. 28*28 = 784
Y = tf.placeholder(tf.float32, [None, 10])  # 0부터 9 까지 10개의 라벨이 필요

# --------------------------- 수정한 부분 ------------------------------ #
# Set model weights
W1 = tf.Variable(tf.random_normal([784, 256]))
W2 = tf.Variable(tf.random_normal([256, 256]))
W3 = tf.Variable(tf.random_normal([256,  10]))

B1 = tf.Variable(tf.random_normal([256]))
B2 = tf.Variable(tf.random_normal([256]))
B3 = tf.Variable(tf.random_normal([ 10]))

# Construct model
L1 = tf.nn.relu(tf.add(tf.matmul(X, W1), B1))
L2 = tf.nn.relu(tf.add(tf.matmul(L1, W2), B2)) # Hidden layer with ReLU activation
hypothesis = tf.add(tf.matmul(L2, W3), B3)     # No need to use softmax here
# ---------------------------- 여기까지 ------------------------------- #

# Minimize error using cross entropy
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))   # softmax loss
# Gradient Descent
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        # 나누어 떨어지지 않으면, 뒤쪽 이미지 일부는 사용하지 않는다.
        total_batch = int(mnist.train.num_examples/batch_size)  # batch_size = 100
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)  # next_batch 함수는 지정한 갯수만큼  image와 label을 순차적으로 반환한다
                                                                    # batch_size 가 100이므로 전체 데이타에서 100개씩 뽑아서 트레이닝을 함
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs, Y: batch_ys})
            # _, 는 버리겠다는 의미이다. cost 함수의 최소값을 찾아내고 필요없는 optimizer 의 값은 버리는 것 같다. 근데 optimizer 를 실행해도 cost가 자동실행되는거 아닌가
            # 분할해서 구동하기 때문에 cost를 계속해서 누적시킨다. 전체 중의 일부에 대한 비용.
            avg_cost += c / total_batch
        # Display logs per epoch step. display_step이 1이기 때문에 if는 필요없다.
        if (epoch+1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", accuracy.eval({X: mnist.test.images, Y: mnist.test.labels}))
