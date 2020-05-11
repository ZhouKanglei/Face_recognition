import os
import sys
import io
import datetime
import numpy as np
np.random.seed(2020) 

from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Flatten
from keras.layers import Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

from get_data_ATT import load_data
from plot_acc_loss import plot_acc_loss

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 全局变量 
DATASET_NAME = 'ATT_2'
MODEL_PATH = 'results/ATT/CNNface/' + DATASET_NAME + '_face_model.h5'
DATASET_PATH = 'data/' + DATASET_NAME
TRAINING_RATE = 0.6
N_CLASSES = 40
BATCH_SIZE = 5
EPOCHS = 50
LEARNING_RATE = 0.001
DECAY = 1e-6

# 输入图片样本的宽高
img_rows, img_cols = 112, 92 
if DATASET_PATH.find('ATT') != -1:
	img_rows, img_cols = 112, 92
	
if DATASET_PATH.find('ATT_1') != -1:
	img_rows, img_cols = 112, 92

if DATASET_PATH.find('ATT_2') != -1:
	img_rows, img_cols = 56, 46

if DATASET_PATH.find('ATT_3') != -1:
	img_rows, img_cols = 224, 224

class Logger(object):
    def __init__(self, filename = 'default.log', stream = sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')
 
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
 
    def flush(self):
        pass

def pre_processing():	
	x_train, y_train, x_vaild, y_vaild, x_test, y_test 	= load_data(
		DATASET_PATH, TRAINING_RATE, N_CLASSES)

	x_train /= 255.0
	x_vaild /= 255.0
	x_test /= 255.0

	# transfer label to binary value
	y_train = to_categorical(y_train, num_classes = N_CLASSES)
	y_vaild = to_categorical(y_vaild, num_classes = N_CLASSES)
	y_test = to_categorical(y_test, num_classes = N_CLASSES)

	print('样本数据集的维度：', x_train.shape, y_train.shape)
	print('检验数据集的维度：', x_vaild.shape, y_vaild.shape)
	print('测试数据集的维度：', x_test.shape, y_test.shape)

	return x_train, y_train, x_vaild, y_vaild, x_test, y_test


def AlexNet_model():
	model = Sequential()  
	model.add(Conv2D(96, (11,11), strides=(4, 4), input_shape = (224, 224, 1), padding='valid', activation = 'relu', kernel_initializer = 'uniform'))  
	model.add(MaxPooling2D(pool_size=(3, 3), strides = (2, 2)))  
	model.add(Conv2D(256, (5, 5),strides=(1, 1), padding='same',activation='relu',kernel_initializer='uniform'))  
	model.add(MaxPooling2D(pool_size=(3, 3), strides = (2, 2)))  
	model.add(Conv2D(384, (3, 3), strides=(1, 1), padding = 'same',activation='relu',kernel_initializer='uniform'))  
	model.add(Conv2D(384, (3, 3), strides=(1, 1), padding = 'same',activation='relu',kernel_initializer='uniform'))  
	model.add(Conv2D(256, (3, 3), strides=(1, 1), padding = 'same',activation='relu',kernel_initializer='uniform'))  
	model.add(MaxPooling2D(pool_size = (3, 3), strides = (2, 2)))  
	model.add(Flatten())  
	model.add(Dense(1024, activation='relu'))  
	model.add(Dropout(0.5))  
	model.add(Dense(512, activation='relu'))  
	model.add(Dropout(0.5))  
	model.add(Dense(40, activation = 'softmax'))  
	
	model.summary()
	return model

def simple_model():
    #搭建网络
    model = Sequential()
    model.add(Conv2D(32, (3, 3), 
        input_shape = (56, 46, 1), 
        padding='same', 
        activation = 'relu'))  
    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))  

    model.add(Flatten()) 

    model.add(Dense(units = 1024, activation = 'relu'))
    model.add(Dense(units = 512, activation = 'relu'))
    model.add(Dense(units = 40, activation = 'softmax'))

    model.summary()
    return model

def build_model():
	# 构建模型
	model = Sequential()

	model.add(Conv2D(32, (3, 3), padding = 'same',
		input_shape = (img_rows, img_cols, 1))) # 卷积层1
	model.add(Activation('relu'))	
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size = (2, 2))) # 池化层1
	model.add(Dropout(0.25))	

	model.add(Conv2D(64, (3, 3), padding = 'same')) # 卷积层2
	model.add(Activation('relu'))
	model.add(BatchNormalization())	
	model.add(Conv2D(64, (3, 3), padding = 'same')) # 卷积层3
	model.add(Activation('relu'))
	model.add(BatchNormalization())	
	model.add(MaxPooling2D(pool_size = (2, 2))) # 池化层2
	model.add(Dropout(0.25))	

	model.add(Conv2D(128, (3, 3), padding = 'same')) # 卷积层4
	model.add(Activation('relu'))
	model.add(BatchNormalization())	
	model.add(Conv2D(128, (3, 3), padding = 'same')) # 卷积层5
	model.add(Activation('relu'))
	model.add(BatchNormalization())	
	model.add(MaxPooling2D(pool_size = (2, 2))) # 池化层3
	model.add(Dropout(0.25))	
	
	# 拉成一维数据
	model.add(Flatten()) 
	model.add(Dense(512))
	model.add(Activation('relu'))
	model.add(BatchNormalization())	
	model.add(Dropout(0.5))

	# softmax 分类器
	model.add(Dense(N_CLASSES)) # 全连接层3
	model.add(Activation('softmax')) # sigmoid评分
	model.summary()

	return model

def model_train(model, x_train, y_train, x_vaild, y_vaild):

	# 数据增强
	aug = ImageDataGenerator(
		rotation_range = 25, 
		width_shift_range = 0.1, 
		height_shift_range = 0.1,
		shear_range = 0.2,
		zoom_range = 0.2,
		horizontal_flip = True,
		fill_mode = 'nearest')

	# 模型编译
	model.compile(loss = 'categorical_crossentropy', 
		optimizer = SGD(lr = LEARNING_RATE, decay = DECAY), 
		metrics = ['accuracy'])

	# 训练模型
	# H = model.fit_generator(
	# 	aug.flow(x_train, y_train, batch_size = BATCH_SIZE), 
	# 	# setps_per_epoch = len(x_train) // batch_size, 
	# 	epochs = EPOCHS, 
	# 	shuffle = True,
	# 	verbose = 1, 
	# 	validation_data = (x_vaild, y_vaild))

	H = model.fit(x_train, y_train, 
		batch_size = BATCH_SIZE, 
		epochs = EPOCHS, 
		# shuffle = True,
		verbose = 1, 
		validation_data = (x_vaild, y_vaild))

	# 保存model
	model.save(MODEL_PATH)

	# 评估模型
	score = model.evaluate(x_vaild, y_vaild)
	print('Vaild loss:', score[0])
	print('Vaild accuracy:', score[1])

	return H

def model_test(x_test, y_test):
	model = load_model(MODEL_PATH)
	model.summary()

	# # 打印结果
	# y = model.predict(x_test)
	# y_test = y_test.argmax(axis = 1) + 1 
	# y_pred = y.argmax(axis = 1) + 1 

	# 评估模型
	score = model.evaluate(x_test, y_test)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])

def main():
	x_train, y_train, x_vaild, y_vaild, x_test, y_test = pre_processing()
	# model = build_model()
	# model = AlexNet_model()
	model = simple_model()
	H = model_train(model, x_train, y_train, x_vaild, y_vaild)
	model_test(x_train, y_train)
	
	# 绘制训练acc vs loss并保存
	plot_acc_loss(H.history['loss'], H.history['accuracy'], 
        H.history['val_loss'], H.history['val_accuracy'])

if __name__ == '__main__':

	daytime = datetime.datetime.now().strftime('%y_%m_%d')
	path = 'results/ATT/CNNface/logs/' + daytime
	if os.path.exists(path + '_log.txt'):
		path + '_log.txt'

	sys.stdout = Logger(path + '_log.txt', sys.stdout)   # 控制台输出日志
	sys.stderr = Logger(path + '_error.txt', sys.stderr) # 错误输出日志
	main()
	