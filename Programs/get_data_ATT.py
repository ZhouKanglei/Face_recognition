#科学计算库
import cv2
import numpy as np


import random
from PIL import Image

from sklearn.model_selection import train_test_split

# from keras import backend as K
# from keras.utils import np_utils

"""
加载图像，读取图像数据
"""

# num_class0个样本，num_class个人，每人10张样本图。
# 每个像素点做了归一化处理
def load_data(dataset_path, rate_train, num_class):

	if dataset_path.find('ATT') != -1:
		type_img = '.pgm'
		height, width = 112, 92
		sum_img = 10

	if dataset_path.find('ATT_3') != -1:
		type_img = '.pgm'
		height, width = 224, 224
		sum_img = 10
	
	if dataset_path.find('ATT_2') != -1:
		type_img = '.pgm'
		height, width = 56, 46
		sum_img = 20

	if dataset_path.find('ATT_1') != -1:
		type_img = '.pgm'
		height, width = 112, 92
		sum_img = 1000

	x = np.empty(shape = (sum_img * num_class, height, width, 1), dtype = 'float')
	y = []

	cur_img = 0
	for i in range(1, num_class + 1):
		for j in range(1, sum_img + 1):
			file_path = dataset_path + '/' + str(i) + '/' + str(j) + type_img
			img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
			x[cur_img, :, :, 0] = img[:]
			y.append(i - 1)
			cur_img += 1

	x_train, x_test, y_train, y_test = train_test_split(x, y, 
		test_size = 1 - rate_train, random_state = 2020)
	x_vaild, x_test, y_vaild, y_test = train_test_split(x_test, y_test, 
		test_size = (1 - rate_train) / 2, random_state = 2021)

	y_train = np.array(y_train)
	y_vaild = np.array(y_vaild)
	y_test = np.array(y_test)

	return x_train, y_train, x_vaild, y_vaild, x_test, y_test

if __name__ == '__main__':
	dataset_path = 'data/ATT'
	rate_train= 0.6
	num_class = 10
	x_train, y_train, x_vaild, y_vaild, x_test, y_test = load_data(
		dataset_path, rate_train, num_class)

	print(train_data.shape)
	print(vaild_data.shape)
	print(test_data.shape)