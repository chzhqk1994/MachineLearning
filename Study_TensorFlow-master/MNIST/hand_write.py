import glob
import numpy
import scipy.special
import scipy.misc

# 손으로 쓴 데이터를 읽어들이는 코드인데 어떻게 쓰는거지 ㅅㅂ

# our own image test data set
our_own_dataset = []

# load the png image data as test data set
for image_file_name in glob.glob('hand/1.png'):
    # use the filename to set the correct label
    label = int(image_file_name[-5:-4])

    # load image data from png files into an array
    print("loading ... ", image_file_name)
    img_array = scipy.misc.imread('hand/1.png', flatten=True)

    # reshape from 28x28 to list of 784 values, invert values
    img_data = 255.0 - img_array.reshape(784)

    # then scale data to range from 0.01 to 1.0
    img_data = (img_data / 255.0 * 0.99) + 0.01
    print(numpy.min(img_data))
    print(numpy.max(img_data))

    # append label and image data  to test data set
    record = numpy.append(label, img_data)
    our_own_dataset.append(record)


    pass
    print(our_own_dataset)
