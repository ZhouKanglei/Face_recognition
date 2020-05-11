
import math
import random
import argparse

#系统库
import os
import sys
import shutil
import random
import time

#科学计算库
import numpy as np
import cv2

#可视化库
from matplotlib import pyplot as plt

class LBPface():
    '''
    实现圆形LBP算子
    圆形区域的半径为R
    采样点数量为P
    '''
    def __init__(self, name_dataset, N_training_img):
        self.N_training_img = N_training_img
        self.N_testing_img = 10 - N_training_img
        self.name_dataset = name_dataset
        self.R = 1
        self.P = self.R * 8

        if self.name_dataset == 'ATT':
            self.height, self.width = 92, 112
            self.dir_identity = 'data/ATT'
            self.N_identity = 40
        if self.name_dataset == 'CASIA-500':
            self.height, self.width = 112, 112
            self.dir_identity = 'data'
            self.N_identity = 500

        self.S = self.height * self.width
        self.N_total_training_img = self.N_identity * self.N_training_img

        #初始化训练图像集
        self.loadImageSet()
        #计算图像的LBP特征
        self.result_LBP = self.cal_LBP(self.imgs_training)

        # 计算所有图像的LBP直方图
        # self.Histograms = np.mat(np.zeros((58 , self.N_total_training_img)))
        self.Histograms = np.mat(np.zeros((58*4*8, self.N_total_training_img)))
        for i in range(self.N_total_training_img):
            self.Histograms[:, i] = np.mat(self.cal_Histogram_img(self.result_LBP[:, i]))
        # print('Cal Histograms down.')


    def loadImageSet(self):
        # Numpy matrix必须是2维的，而array可以是多维的，matrix包含于array
        self.imgs_training = np.zeros((self.S, self.N_total_training_img), dtype='float64')

        if self.name_dataset == 'ATT':
            self.type_img = '.pgm'
        if self.name_dataset == 'CASIA-500':
            self.type_img = '.png'

        self.ids_training = []
        index_img = 0

        for id_face in range(1, self.N_identity + 1):
            # 产生N个训练图像的list
            ids_img = random.sample(range(1, 11), self.N_training_img)
            self.ids_training.append(ids_img)

            # if id_face == 1:
            #     path_img = os.path.join(self.dir_identity, str(id_face), str(ids_img[0]) + self.type_img)

            #     # print('> reading file: ' + path_img)
            #     img = cv2.imread(path_img, 0)
            #     cv2.imshow('Orignal_window', img)
            #     cv2.waitKey(0)
            #     cv2.destroyAllWindows()

            for id_img in ids_img:
                path_img = os.path.join(self.dir_identity, str(id_face), str(id_img) + self.type_img)

                # print('> reading file: ' + path_img)
                img = cv2.imread(path_img, 0)
                vector_img = np.array(img, dtype = 'float64').flatten()
                # 加到总的imgs_training中去, 行向量，之前都是列向量,这里的行向量可能有点问题
                self.imgs_training[:, index_img] = vector_img[:]
                index_img += 1
                # print(self.imgs_training[:, index_img])


    def minBinary(self ,str_b):
        '''
        :param str_b: 一个二进制的字符串，是LBP的初始值
        :return: 最小的LBP值
        '''
        min_s = int(str_b, base=2)
        str_min = ''
        for i in range(len(str_b)):
            str_b = str_b[-1] + str_b[:-1]
            temp = int(str_b, base=2)
            if temp < min_s:
                str_min = str_b
                min_s = temp
        #print(min_s)
        return min_s

    def find_variations(self, str_b):
        prev = str_b[-1]
        t = 0
        for i in range(0, len(str_b)):
            cur = str_b[i]
            if cur != prev:
                t += 1
            prev = cur
        return t

    def cal_LBP(self, imgs_trainging):
        #x,y是采样点坐标
        imgs_trainging = imgs_trainging
        #Region8_x, Region8_y = [-1, 0, 1, 1, 1, 0, -1, -1], [-1, -1, -1, 0, 1, 1, 1, 0]
        pi = math.pi
        result_LBP = np.mat(np.zeros(np.shape(imgs_trainging)))
        for i in range(0, np.shape(imgs_trainging)[1]):
            # 处理每一个图像
            # print('Cal %d LBP down...'%i)
            img_face = imgs_trainging[:, i].reshape(self.width, self.height)
            tempface = np.mat(np.zeros((self.width, self.height), dtype='float64'))

            for x in range(self.R, self.width - self.R):
                for y in range(self.R, self.height - self.R):
                    # 每一个点都需要计算一次LBP值
                    value_LBP_orginal = ''

                    value_0_0 = int(img_face[x, y])

                    for p in range(1, self.P + 1):
                        # 获取采样点坐标，上面的list顺序是没关系的
                        p = float(p)
                        xp = x + self.R * math.cos(2 * pi * (p / self.P))
                        yp = y - self.R * math.sin(2 * pi * (p / self.P))

                        if int(xp) == xp:
                            if int(yp) == yp:
                                value_xp_yp = value_0_0
                            else:
                                y_down = math.ceil(yp)
                                y_up = int(yp)
                                w1 = (y_down-yp)/(y_down-y_up)
                                w2 = (yp-y_up)/(y_down-y_up)
                                value_xp_yp = (w1*int(img_face[int(xp), int(y_up)]) + w2*int(img_face[int(xp), int(y_down)]))/(w1+w2)
                                value_xp_yp = int(value_xp_yp)
                        elif int(yp) == yp:
                            x_right = math.ceil(xp)
                            x_left = int(xp)
                            w1 = (x_right - xp) / (x_right - x_left)
                            w2 = (xp - x_left) / (x_right - x_left)
                            value_xp_yp = (w1 * int(img_face[x_left, int(yp)]) + w2 * int(img_face[x_right, int(yp)])) / (w1 + w2)
                            value_xp_yp = int(value_xp_yp)
                        else:
                            #双线性差值求该点的灰度值，注意如果xp, yp正好是整数，不能使用双线性差值
                            x_left = int(xp)
                            x_right = math.ceil(xp)
                            y_up = int(yp)
                            y_down = math.ceil(yp)

                            value_left_up = img_face[x_left, y_up]
                            value_left_down = img_face[x_left, y_down]
                            value_right_up = img_face[x_right, y_up]
                            value_right_down = img_face[x_right, y_down]
                            #事实上right-left=1
                            value_xp_down = (x_right-xp)/1*value_left_down + (xp-x_left)/1*value_right_down
                            value_xp_up = (x_right-xp)/1*value_left_up+(xp-x_left)/1*value_right_up

                            value_xp_yp = (y_up-yp)/(-1)*value_xp_down+(yp-y_down)/(-1)*value_xp_up


                            #value_xp_yp = img_face[round(xp), round(yp)]
                        if value_xp_yp > value_0_0:
                            value_LBP_orginal += '1'
                        else:
                            value_LBP_orginal += '0'

                    #tempface[x-1, y-1] = self.minBinary(value_LBP_orginal)
                    #tempface[x, y] = int(value_LBP_orginal, base=2)
                    tempface[x, y] = self.find_variations(value_LBP_orginal)
         
            # if i == 1:
            #     cv2.imshow('Test_window', tempface)
            #     cv2.waitKey(0)
            #     cv2.destroyAllWindows()
            
            result_LBP[:, i] = tempface.flatten().T

        # print('Cal LBP down...')
        return result_LBP

    def cal_Histogram_img(self, LBP_img, plot = False):
        '''
        输入单个图像的LBP特征
        :return: LBP直方图
        '''
        LBP_img = LBP_img.reshape(self.width, self.height)
        #LBP有256种取值，更合适的应该是取成5种，每50是一种，利用等价模式变成了58种
        #把图像分为8*4份
        #Historgram = np.mat(np.zeros((58, 8 * 4)))
        Historgram = np.mat(np.zeros((58, 8*4)))
        maskx, masky = int(self.width/8), int(self.height/4)
        for i in range(8):
            for j in range(4):
                mask = np.zeros((self.width, self.height), dtype='uint8')
                #mask就是需要处理的图像部分
                mask[i*maskx:(i+1)*maskx, j*masky:(j+1)*masky] = 1
                LBP_img = np.array(LBP_img, dtype='uint8')
                
                # if plot == True:
                #     hist, bin = np.histogram(LBP_img.flatten(), 256, [0, 256])
                #     #可视化直方图
                #     plt.hist(LBP_img.flatten(), 256, [0, 256])
                #     plt.xlim([0, 256])
                #     plt.legend(('cdf', 'histogram'), loc='upper left')
                #     plt.show()
                
                #hist = cv2.calcHist([LBP_img], [0], None, [58], [0, 255])
                #return hist.reshape(58, -1)
                hist = cv2.calcHist([LBP_img], [0], mask, [58], [0, 255])
                #Historgram一共有32列,对应32个区域的直方图分布

                Historgram[:, (i+1)*(j+1)-1] = np.mat(hist).flatten().T
                #print('Cal %d,%d hist down'%(i, j))

        return Historgram.flatten().T


    def find_indentity(self, path_to_img):
        img = cv2.imread(path_to_img, 0)

        img_vector = np.array(img, dtype = 'float64').flatten()
        imgs_testing = np.zeros((self.S, 1), dtype='float64')
        imgs_testing[:, 0] = img_vector
        LBP_img = self.cal_LBP(imgs_testing)
        Histogram_img = self.cal_Histogram_img(LBP_img)
        # print(Histogram_img.T)
        # index_min = 0
        # value_min = np.inf
        # distance = np.mat(np.zeros(shape=(58, self.N_total_training_img)))
        distance = np.mat(np.zeros(shape = 
            (Histogram_img.shape[0], self.N_total_training_img)))
        for i in range(self.N_total_training_img):
            
            # diff = (np.array(self.Histograms[:, i]-Histogram_img[:])**2).sum()
            # if diff < value_min:
            #     index_min = i
            #     value_min = diff
            
            distance[:, i] = self.Histograms[:, i] - Histogram_img[:]
            
        # print(index_min)
        
        norms = np.linalg.norm(distance, axis = 0)
        closet_face_id = np.argmin(norms)
        # print(closet_face_id)
        return int(closet_face_id / self.N_training_img) + 1

def parse_args():
    parser = argparse.ArgumentParser(description='MyLBP parameters')
    # general
    parser.add_argument('--name_dataset', default='ATT',
                        help='name of dataset, including att and CASIA-500')
    #parser.add_argument('--dir_data', default='data/ATT', help='training and testing set directory')
    #parser.add_argument('--dir_data', default='data',
    #                    help='training and testing set directory')
    #parser.add_argument('--size_img', default='112*112', help='image size')
    #parser.add_argument('--N_identity', default=500, help='Numbers of dataset person')
    parser.add_argument('--N_training_img', default=3,
                        help='Number of person for training in a indentity, rest is test')

    args = parser.parse_args()

    return args

def verification(sample_LBP):
    
    cnt = 0
    sum_num = 0
    for i in range(1, 41):
        ids_img = random.sample(range(1, 11), 10 - sample_LBP.N_training_img)
        for j in ids_img:
            face_id = 'data/ATT/%d/%d.pgm'%(i, j)
            result_id = sample_LBP.find_indentity(face_id)
            sum_num += 1
            if result_id == i:
                cnt += 1
            # print('Seach_id %s, result_id is %d' % (face_id, result_id))

    print('Correct rate: %.2f'%(1.0 * cnt / sum_num * 100))
    return 1.0 * cnt / sum_num * 100

def plot_fig(face_dataset, time_train, 
        time_test, Accuracy, N):
    plt.plot(Accuracy, '-bo')
    plt.plot(np.array(Accuracy) - np.array(Accuracy)
     + np.mean(Accuracy), '-g',
     label = 'Mean Accuracy = %.4f%%' 
        % np.mean(Accuracy))
    plt.plot(Accuracy.index(np.max(Accuracy)), np.max(Accuracy), 
        'ro', label = 'Best Accuracy = %.4f%%' 
        % np.max(Accuracy))
    plt.title(
        'Recognition Accuracy versus $\mathcal{T}$\n')
    plt.xlabel('$\mathcal{T}$: number of times')
    plt.ylabel('Recognition Accuracy (%)')
    plt.legend()
    fig_name = 'results/' + face_dataset + '/LBPface/accuracy_versus_times.png'
    plt.savefig(fig_name, dpi = 200, transparent = True)
    plt.figure()

    plt.plot(time_train, '-ro', label = 'Train Time')
    plt.plot(time_test, '-bo', label = 'Test Time')
    plt.plot(np.array(time_train) - np.array(time_train)
     + np.mean(time_train), '-g',
     label = 'Mean Train Time = %.4f sec' 
        % np.mean(time_train))
    plt.plot(np.array(time_test) - np.array(time_test)
     + np.mean(time_test), '--g',
     label = 'Mean Test Time = %.4f sec' 
        % np.mean(time_test))
    plt.title(
        'Execution Time versus $\mathcal{T}$\n')
    plt.xlabel('$\mathcal{T}$: number of times')
    plt.ylabel('Execution Time (sec)')
    plt.legend()
    fig_name = 'results/' + face_dataset + '/LBPface/time_versus_times.png'
    plt.savefig(fig_name, dpi = 200, transparent = True)


def main(args):
    Accuracy = []
    time_train = []
    time_test = []
    R = []
    N = []

    for i in range(1, 11):
        start_train = time.time()
        sample_LBP = LBPface(args.name_dataset, args.N_training_img)
        end_train = time.time()
        time_train.append(end_train - start_train)
        print('Train End...', end = '')

        start_test = time.time()
        corr = verification(sample_LBP)
        end_test = time.time()
        time_test.append(end_test - start_test)

        Accuracy.append(corr)
        R.append(sample_LBP.R)
        N.append(args.N_training_img)

    plot_fig(args.name_dataset, time_train, 
        time_test, Accuracy, N)


    print('Times\tTrain Time (sec)\tTest Time (sec)\tAccuary (%)\tRadius\tTrain Images (per capita)')
    for i in range(10):
        print('%d\t%.4f\t%.4f\t%.4f\t%d\t%d'%(
            i + 1, time_train[i], time_test[i],
            Accuracy[i], R[i], N[i]))

    print('Average\t%.4f\t%.4f\t%.4f\t%.4f\t%d'%(
        np.mean(time_train), np.mean(time_test), 
        np.mean(Accuracy), np.mean(R), np.mean(N)))

if __name__ == '__main__':

    args = parse_args()
    main(args)
