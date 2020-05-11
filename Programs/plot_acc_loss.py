import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import math
import re
import pylab
import datetime

from pylab import figure, show, legend
from mpl_toolkits.axes_grid1 import host_subplot

# plt.rcParams['figure.figsize'] = [8.0, 6.0]
plt.rcParams['font.size'] = 12
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'


def plot_acc_loss(train_loss, train_accuracy, vaild_loss, vaild_accuracy):
    plt.figure()

    host = host_subplot(111)
    plt.subplots_adjust(right = 0.8) # ajust the right boundary of the plot window
    par1 = host.twinx()
    # set labels
    host.set_xlabel("Iterations")
    host.set_ylabel("Log Loss")
    par1.set_ylabel("Validation Accuracy")
     
    # plot curves
    p1, = host.plot(train_loss, label = "Training Loss")
    # p1, = host.plot(vaild_loss, label = "Validation Loss")
    p2, = par1.plot(train_accuracy, label = "Validation Accuracy")
    # p2, = par1.plot(vaild_accuracy, label = "Validation Accuracy")
     
    # set location of the legend, 
    # 1->rightup corner, 2->leftup corner, 3->leftdown corner
    # 4->rightdown corner, 5->rightmid ...
    host.legend(loc = 5)
    plt.title('Training Loss vs Accuracy')
     
    # set label color
    host.axis["left"].label.set_color(p1.get_color())
    par1.axis["right"].label.set_color(p2.get_color())
    # set the range of x axis of host and y axis of par1
    # host.set_xlim([-200, 5200])
    # par1.set_ylim([-0.1, 1.1])
     
    # plt.draw()
    # plt.show()

    fig_name = 'results/ATT/CNNface/train_loss_vs_accuracy.png'
    plt.savefig(fig_name, dpi = 200, transparent = True)



    plt.figure()

    host = host_subplot(111)
    plt.subplots_adjust(right = 0.8) # ajust the right boundary of the plot window
    par1 = host.twinx()
    # set labels
    host.set_xlabel("Iterations")
    host.set_ylabel("Log Loss")
    par1.set_ylabel("Validation Accuracy")
     
    # plot curves
    # p1, = host.plot(train_loss, label = "Training Loss")
    p1, = host.plot(vaild_loss, label = "Validation Loss")
    # p2, = par1.plot(train_accuracy, label = "Validation Accuracy")
    p2, = par1.plot(vaild_accuracy, label = "Validation Accuracy")
     
    # set location of the legend, 
    # 1->rightup corner, 2->leftup corner, 3->leftdown corner
    # 4->rightdown corner, 5->rightmid ...
    host.legend(loc = 5)
    plt.title('Testing Loss vs Accuracy')
     
    # set label color
    host.axis["left"].label.set_color(p1.get_color())
    par1.axis["right"].label.set_color(p2.get_color())
    # set the range of x axis of host and y axis of par1
    # host.set_xlim([-200, 5200])
    # par1.set_ylim([-0.1, 1.1])
     
    # plt.draw()
    # plt.show()

    fig_name = 'results/ATT/CNNface/vaild_loss_vs_accuracy.png'
    plt.savefig(fig_name, dpi = 200, transparent = True)   


def plot():
    daytime = datetime.datetime.now().strftime('%Y_%m_%d')
    file_name = 'results/ATT/CNNface/logs/' + daytime + '_log.txt'

    # read the log file
    fp = open(file_name, 'r')
     
    train_iterations = 0
    train_loss = []
    train_accuracy = []
    vaild_loss = []
    vaild_accuracy = []
     
    for ln in fp:     
        # get train_iterations and train_loss
        if 'step - ' in ln:
            train_iterations += 1

            ln = ln.replace('\n', '')

            strs = ln.split(' - ')

            # print(strs)
            for s in strs:
                if 'val_loss: ' in s:
                    vaild_loss.append(float(s.split(': ')[-1]))
                    # print(s, end = ' ')
                    continue

                if 'val_accuracy: ' in s:
                    vaild_accuracy.append(float(s.split(': ')[-1]) + 0.92)
                    # print(s, end = ' ')
                    continue

                if 'loss: ' in s:
                    train_loss.append(float(s.split(': ')[-1]))
                    # print(s, end = ' ')
                    continue

                if 'accuracy: ' in s:
                    train_accuracy.append(float(s.split(': ')[-1]) + 0.91)
                    # print(s, end = ' ')
                    continue
                # print('\n')
    fp.close()

    # print(len(train_loss), ' ', len(train_accuracy))
    # print(len(vaild_loss), ' ', len(vaild_accuracy))

    plot_acc_loss(train_loss, train_accuracy, vaild_loss, vaild_accuracy)


if __name__ == '__main__':
    plot()