import numpy as np

# #x0 x1 x2 y
#  1  1  0  1
#  1  0  2  2
#  1  3  0  3
#  1  0  4  4
#  1  5  0  5

data1 = np.loadtxt('train.txt', dtype='float32')  # 전치시키지 않은 상태의 첫번째 열
print(data1.shape)
print(data1[0])
test_data1 = np.transpose(data1) # unpack = True 와 같은 역할(전치시킴)
print(test_data1)
print("\n")


# x0   1 1 1 1 1
# x1   1 0 3 0 5
# x2   0 2 0 4 0
# y    1 2 3 4 5

data2 = np.loadtxt('train.txt', unpack=True, dtype = 'float32' ) # 전치시킨 상태의 첫번째 열
print(data2.shape)
print(data2[0])
test_data2 = np.transpose(data2) # unpack = True 와 같은 역할(전치시킴)
print(test_data2)

# x_data = np.transpose(xy[0:3])
# y_data = np.transpose(xy[3:])