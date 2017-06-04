import tensorflow as tf

class Linear_regression:
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

        self.W = tf.Variable(tf.random_uniform([1],-1.0, 1.0))
        self.b = tf.Variable(tf.random_uniform([1],-1.0, 1.0))

        self.X = tf.placeholder(tf.float32)
        self.Y = tf.placeholder(tf.float32)

        self.Hypothesis()

    def Hypothesis(self):
        self.hypothesis = self.W * self.X + self.b
        self.cost = tf.reduce_mean(tf.square(self.hypothesis - self.Y))


    def Learning(self,Learning_rate):   #Learning rate 을 입력받 을 수있음
        self.a = tf.Variable(Learning_rate)
        self.optimizer = tf.train.GradientDescentOptimizer(self.a)
        self.train = self.optimizer.minimize(self.cost)

        self.initialize()

    def initialize(self):
        self.init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(self.init)

    def output(self, step):
        for self.step in range(step):
            self.sess.run(self.train, feed_dict={self.X: self.x_data, self.Y: self.y_data})
            if self.step % 20 == 0:
                print(self.step, self.sess.run(self.cost, feed_dict={self.X: self.x_data, self.Y: self.y_data}), self.sess.run(self.W), self.sess.run(self.b))

# lr = Linear_regression([1,2,3],[1,2,3])
# lr.Hypothesis()
# lr.Learning(0.1)
# for step in range(2001):
#     lr.sess.run(lr.train, feed_dict={lr.X: lr.x_data, lr.Y : lr.y_data})
#     if step%20==0:
#         print(step, lr.sess.run(lr.cost, feed_dict={lr.X: lr.x_data, lr.Y: lr.y_data}), lr.sess.run(lr.W), lr.sess.run(lr.b))