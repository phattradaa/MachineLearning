from sklearn import datasets
import pylab
import matplotlib.pyplot as plt
from scipy.io import loadmat
#IRIS DATASET
iris_dataset = datasets.load_iris()
# print(iris_dataset.keys())
# print(iris_dataset['data'][0:10])

#MNIST DATASET
digit_dataset = datasets.load_digits()
# print(digit_dataset.images[0])
# print(digit_dataset.images[0].shape)
# print(digit_dataset.target_names[2])

#Show Dataset by Pylab 
pylab.imshow(digit_dataset.images[2],cmap=pylab.cm.gray_r)
# pylab.show()

#Show Dataset By Matplotlib
plt.imshow(digit_dataset.images[3],cmap=plt.get_cmap('gray'))
# plt.show()


mnist_raw = loadmat("mnist-original.mat")
mnist={
    "data":mnist_raw["data"].T,
    "target":mnist_raw["label"][0]
}
print(mnist["data"].shape)