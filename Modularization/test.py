from MachineLearning.Modularization import Module_linear_regression as lr

x_data = [1,3,5,7,9]
y_data = [2,4,6,8,10]

lr = lr.Linear_regression(x_data, y_data)
lr.Learning(0.1)
lr.output(2001)
# lr = Linear_regression([1,2,3],[1,2,3])
# lr.Hypothesis()
# lr.Learning(0.1)
# for step in range(2001):
#     lr.sess.run(lr.train, feed_dict={lr.X: lr.x_data, lr.Y : lr.y_data})
#     if step%20==0:
#         print(step, lr.sess.run(lr.cost, feed_dict={lr.X: lr.x_data, lr.Y: lr.y_data}), lr.sess.run(lr.W), lr.sess.run(lr.b))