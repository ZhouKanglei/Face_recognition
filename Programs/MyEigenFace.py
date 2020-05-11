import argparse

#系统库
import os
import sys
import shutil
import random
import time

#科学计算库
import cv2
import numpy as np

from Visualize import plot_confusion_matrix

import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# plt.rcParams['figure.figsize'] = [8.0, 6.0]
plt.rcParams['font.size'] = 12
# plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

class Eigenface():

    #初始化模型参数
    def __init__(self, face_dataset, total_faces_n=40,img_size='92*112', energy=0.85):
        #energy应该是阈值的意思
        self.face_dataset = face_dataset
        self.faces_dir = 'data/' + face_dataset
        self.energy = energy

        if self.faces_dir.find('ATT') != -1:
            self.type_img = '.pgm'
        else:
            self.type_img = '.png'
        self.total_faces_n = total_faces_n
        self.training_faces_cout = 6
        self.test_faces_cout = 10-self.training_faces_cout
        #6*40, 40个人，每一个人取6张图片训练
        self.total_train_faces = total_faces_n * self.training_faces_cout
        self.img_hight, self.img_width = int(img_size.split('*')[0]), int(img_size.split('*')[1])
        self.training_ids_total = []
        cur_img = 0
        # 每一个人的训练图像集
        training_ids = []

        #用于训练的所有人脸列
        imgs_training_total = np.empty(shape=(self.img_width*self.img_hight, self.total_train_faces), dtype='float64')


        #每一个人face_id
        for face_id in range(1, self.total_faces_n+1):
            #在1到10中随机取9个，为什么是随机的呢
            training_ids = random.sample(range(1, 11), self.training_faces_cout)
            self.training_ids_total.append(training_ids)

            #每一张单张的人脸
            for training_id in training_ids:
                path_img = os.path.join(self.faces_dir, str(face_id), str(training_id)+self.type_img)

                print('> reading file: '+ path_img)

                img = cv2.imread(path_img, 0)#灰度图
                # print(img.shape)
                #把图像长方形拉成一列
                img_col = np.array(img, dtype='float64').flatten()
                #加到total training imgs里面去
                # print(type(img_col[:]))
                imgs_training_total[:, cur_img] = img_col[:]
                cur_img += 1

        #已经得到了完整的训练集, 然后求平均脸
        self.mean_img_col = np.sum(imgs_training_total, axis=1) / self.total_train_faces

        for col in range(0, self.total_train_faces):
            imgs_training_total[:, col] -= self.mean_img_col[:]

        #协方差矩阵C
        C = np.matrix(imgs_training_total.transpose()) * np.matrix(imgs_training_total)

        #求列平均
        C /= self.total_train_faces

        self.matrix = C

        self.cal_eva_evc(C)
        # 再imgs_training_total×特征向量，才是真正的特征向量
        self.evectors = imgs_training_total * self.evectors
        #归一化
        norms = np.linalg.norm(self.evectors, axis=0)
        self.evectors = self.evectors / norms

        #人脸的权重向量W， Ω
        print('Cal the weight of face')
        self.W = self.evectors.transpose() * imgs_training_total

    def save_matrix(self, C, cnt):
        plt.figure(figsize = (12, 8))
        #二维的数组的热力图，横轴和数轴的ticklabels要加上去的话，既可以通过将array转换成有column
        #和index的DataFrame直接绘图生成，也可以后续再加上去。后面加上去的话，更灵活，包括可设置labels大小方向等。
        sns.heatmap(C)
        #sns.heatmap(np.round(a,2), annot=True, vmax=1,vmin = 0, xticklabels= True, yticklabels= True, 
        #            square=True, cmap="YlGnBu")
        plt.title('Covariance Matrix Heat Map')

        fig_name = 'results/' + self.face_dataset + '/Eigenface/heatmap.png'
        plt.savefig(fig_name, dpi = 1000, transparent = True)

        plt.figure()
        length = len(self.evalues)
        x_1 = np.linspace(0, cnt - 1, num = cnt)
        y_1_low = x_1 - x_1
        y_1_high = self.evalues[0:cnt]
        x_2 = np.linspace(cnt, length - 1, num = length - cnt)
        y_2_low = x_2 - x_2
        y_2_high = self.evalues[cnt:length]
        y_3 = np.linspace(0, np.max(self.evalues), num = cnt)
        x_3 = y_3 - y_3 + cnt - 1 
        plt.plot(self.evalues)
        plt.fill_between(x_1, y_1_high, 
            y_1_low, facecolor = 'green', alpha = 1, label = 'Important Components')
        plt.fill_between(x_2, y_2_high, 
            y_2_low, facecolor = 'yellow', alpha = 1, label = 'Unimportant Components')
        plt.plot(x_3, y_3, 'r--', label = 'Threshold Line')
        plt.title('Eigenvalue Arrangement')

        plt.legend()
        fig_name = 'results/' + self.face_dataset + '/Eigenface/eigenvals.png'
        plt.savefig(fig_name, dpi = 200, transparent = True)

    def cal_eva_evc(self, C):
        # 求协方差矩阵的特征值和特征向量
        
        self.evalues, self.evectors = np.linalg.eig(C)
        #indices返回从大到小的索引值
        sort_indices = self.evalues.argsort()[::-1]
        self.evalues = self.evalues[sort_indices]
        self.evectors = self.evectors[:, sort_indices]
        

        evalues_sum = sum(self.evalues[:])
        evalues_cout = 0
        evalues_energy = 0.0#阈值?权重

        for evalue in self.evalues:
            evalues_cout += 1
            evalues_energy += evalue / evalues_sum

            #特征值大小的意义是什么？
            #特征值越大，特征向量越能描述原矩阵吗？
            if evalues_energy >= self.energy:
                break

        #剪枝特征向量和特征值, 找到主成分，其实就是PCA的原理
        # self.save_matrix(C, evalues_cout)
        self.evalues = self.evalues[0:evalues_cout]
        self.evectors = self.evectors[:,0:evalues_cout]

    def search_in_dataset(self, path_to_img):
        img = cv2.imread(path_to_img, 0)
        #把人脸也拉成一列
        img_col = np.array(img, dtype='float64').flatten()
        #减去平均脸
        img_col -= self.mean_img_col
        img_col = np.reshape(img_col, (self.img_width*self.img_hight, 1))
        #将人脸投影到特征向量上

        S = self.evectors.transpose() * img_col

        diff = self.W - S
        #计算距离的模
        norms = np.linalg.norm(diff, axis=0)

        #找到距离最小的那一列
        closet_face_id = np.argmin(norms)
        #找到face_id
        return int(closet_face_id/self.training_faces_cout) +1

    def verification(self):
        #可用10折交叉验证，来评估模型的准确率
        #从训练集中分割出的每人4张人脸，与训练集是同分布的
        print('> Evaluation ', self.face_dataset, ' faces started')
        log_name = self.face_dataset + '/Eigenface/results.txt'
        results_file = os.path.join('results', log_name)
        #新建文件夹
        if not os.path.exists('results'):
            os.makedirs(results_file)
        test_count = self.test_faces_cout * self.total_faces_n
        test_correct = 0
        with open(results_file, 'w+') as f:
            for face_id in range(1, self.total_faces_n+1):
                #用没有用来测试的图像来做验证
                for test_id in range(1, 11):
                    if test_id not in self.training_ids_total[face_id-1]:
                        path_to_img = os.path.join(self.faces_dir, str(face_id), str(test_id)+self.type_img)
                        result_id = self.search_in_dataset(path_to_img)
                        result = (result_id == face_id)

                        if result == True:
                            test_correct += 1
                            f.write('image: %s\nresult: correct\n\n'%(path_to_img))
                        else:
                            f.write('image: %s\nresult: wrong, got %2d\n\n'%(path_to_img, result_id))

            print('> Evaluating ATT faces ended')
            self.accuracy = float(100. * test_correct/test_count)
            print('Correct: '+ str(self.accuracy) + '%')
            f.write('Correct: %.4f\n' %(self.accuracy))

def plot_fig(face_dataset, time_train, 
        time_test, Accuracy, N):
    plt.figure()
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
    fig_name = 'results/' + face_dataset + '/Eigenface/accuracy_versus_times.png'
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
    fig_name = 'results/' + face_dataset + '/Eigenface/time_versus_times.png'
    plt.savefig(fig_name, dpi = 200, transparent = True)


def main(args):
    #初始化模型
    Accuracy = []
    time_train = []
    time_test = []
    N = []
    since = time.time()
    eigenvals = []
    for i in range(0, 10):
        start_train = time.time()
        EigenFace = Eigenface(args.face_dataset, args.total_faces_n, args.img_size)
        end_train = time.time() 
        time_train.append(end_train - start_train)
        #得到人脸权重向量
        #在训练集的同分布验证集上测试结果
        start_test = time.time()
        EigenFace.verification()
        end_test = time.time()
        time_test.append(end_test - start_test)

        Accuracy.append(EigenFace.accuracy)
        N.append(EigenFace.training_faces_cout)

        eigenvals.append(len(EigenFace.evalues))

    plot_fig(args.face_dataset, time_train, 
        time_test, Accuracy, N)

    print('Times\tTrain Time (sec)\tTest Time (sec)\tAccuary (%)\tNumber of Eigenvalues\tTrain Images (per capita)')
    for i in range(10):
        print('%d\t%.4f\t%.4f\t%.4f\t%d\t%d'%(
            i + 1, time_train[i], time_test[i],
            Accuracy[i], eigenvals[i], N[i]))

    print('Average\t%.4f\t%.4f\t%.4f\t%.4f\t%d'%(
        np.mean(time_train), np.mean(time_test), 
        np.mean(Accuracy), np.mean(eigenvals), np.mean(N)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='My implement of Eigenface face recognition')
    parser.add_argument('--face_dataset', type=str, default='ATT',
                       help='train image root')
    parser.add_argument('--total_faces_n', type=int, default=40,
                        help='the number of identity')
    parser.add_argument('--img_size', type=str, default='112*92',
                        help='the size of image')
    parser.add_argument('--method', type=str, default='Eigenface',
                        help='method algorithm')
    #parser.add_argument('--test_dataset', type=str, default='celebrity_faces', help='test image root')

    args = parser.parse_args()
    main(args)
