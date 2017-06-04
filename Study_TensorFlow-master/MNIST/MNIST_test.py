import tensorflow as tf

# Import MINST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True) # 인터넷에서 가져온 데이터셋은 /tmp/data 에 저장한다

# Parameters. 반복문에서 사용하는데, 미리 만들어 놓았다.
learning_rate = 0.1
training_epochs = 25

#분류할 이미지 수를 정한다
batch_size = 100
display_step = 1

# tf Graph Input
x = tf.placeholder(tf.float32, [None, None]) # mnist data image of shape 28*28=784
y = tf.placeholder(tf.float32, [None, 10])  # 0-9 digits recognition => 10 classes

# Set model weights
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Construct model
activation = tf.nn.softmax(tf.matmul(x, W) + b)  # Softmax (hytothesis)

# Minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(activation), reduction_indices=1))
# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# 그래프 실행
with tf.Session() as sess:
    sess.run(init)

    # 학습 과정
    # for epoch in range(training_epochs): # trainint_epochs = 25, batch_size = 100
    #     avg_cost = 0.
    #     # 나누어 떨어지지 않으면, 뒤쪽 이미지 일부는 사용하지 않는다.
    #     total_batch = int(mnist.train.num_examples/batch_size)
    #     # 모든 batch에 대해 반복
    #     for i in range(total_batch):
    #         batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    #         # Run optimization op (backprop) and cost op (to get loss value)
    #         _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs, y: batch_ys}) # batch 데이터를 사용해 트레이닝. 연산 후 optimizer 값은 버린다 ( _, )
    #         # 분할해서 구동하기 때문에 cost를 계속해서 누적시킨다. 전체 중의 일부에 대한 비용.
    #         avg_cost += c / total_batch
    #
    #     # 각 반복 단계마다 로그 출력. display_step이 1이기 때문에 if는 필요없다.
    #     if (epoch+1) % display_step == 0:
    #         print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost)) # cost값 확인


    for i in range(1000): # 실제로 학습시키는 코드는 이거믄 되는데 위에는 뭐가 저렇게 많은거냐 시발
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})


    # 추가한 코드. Label과 Prediction이 같은 값을 출력하면 맞는 것이다.
    import random
    r = random.randrange(mnist.test.num_examples)
    print('Label : ', sess.run(tf.argmax(mnist.test.labels[r:r+1], 1)))
    print('Prediction :', sess.run(tf.argmax(activation, 1), {x: mnist.test.images[r:r+1]}))

    # 1줄로 된 것을 28x28로 변환
    import matplotlib.pyplot as plt
    plt.imshow(mnist.test.images[r:r+1].reshape(28, 28), cmap='Greys', interpolation='nearest') # 이미지를 출력
    plt.show()

    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(activation, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

print('r: ', r)