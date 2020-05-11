import numpy as np
import cv2
import os
import pandas
import random
from PIL import Image

#缩放，输入文件名，输出文件名，放大高与宽，偏离度
def resizeImg(img_file1, out_file, dstWeight, dstHeight, deviation):
    img1 = cv2.imread(img_file1, cv2.IMREAD_GRAYSCALE)
    imgshape = img1.shape
     
    h = imgshape[0]
    w = imgshape[1]
    final_matrix = np.zeros((h, w), np.uint8)
    x1 = 0
    y1 = h
    x2 = w
    y2 = 0  #图片高度，坐标起点从上到下  
    dst = cv2.resize(img1, (dstWeight, dstHeight))
    if h < dstHeight:
        final_matrix[y2:y1, x1:x2] = dst[y2+deviation:y1+deviation , x1+deviation:x2+deviation]
    else:
        if deviation == 0:
          final_matrix[y2:dstHeight, x1:dstWeight] = dst[y2:dstHeight, x1:dstWeight]
        else:
          final_matrix[y2 + deviation:dstHeight + deviation, x1 + deviation:dstWeight + deviation] = dst[y2 :dstHeight,x1 :dstWeight]
    cv2.imwrite(out_file, dst)
       
    return final_matrix


#旋转图像，输入文件名、输出文件名，旋转角度
def rotationImg(img_file1, out_file, ra):
    # 获取图片尺寸并计算图片中心点
    img = cv2.imread(img_file1, cv2.IMREAD_GRAYSCALE)
    (h, w) = img.shape[:2]
    center = (w/2, h/2)
    
    M = cv2.getRotationMatrix2D(center, ra, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h))
    #cv2.imshow("rotated", rotated)
    #cv2.waitKey(0)
    cv2.imwrite(out_file, rotated)
     
    return rotated


# 添加椒盐噪声，prob:噪声比例 
def sp_noiseImg(img_file1, out_file,  prob):
    image = cv2.imread(img_file1, cv2.IMREAD_GRAYSCALE)
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob 
    for i in range(image.shape[0]):
      for j in range(image.shape[1]):
        rdn = random.random()
        if rdn < prob:
          output[i][j] = 0
        elif rdn > thres:
          output[i][j] = 255
        else:
          output[i][j] = image[i][j]
    
    cv2.imwrite(out_file, output)
    return output

# 添加高斯噪声
# mean : 均值
# var : 方差
def gasuss_noiseImg(img_file1, out_file, mean = 0, var = 0.002):
    image = cv2.imread(img_file1, cv2.IMREAD_GRAYSCALE)
    image = np.array(image/255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
      low_clip = -1.
    else:
      low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out*255)
    cv2.imwrite(out_file, out)
     
    return out

def noise():
    path = 'data/ATT_1/'
    for i in range(1, 41):
      for j in range(1, 11):
        img_file = path + str(i) + '/' + str(j) + '.pgm'
        out_file = path + str(i) + '/' + str(j + 10) + '.pgm'
        sp_noiseImg(img_file,  out_file,  0.01)

    for i in range(1, 41):
      for j in range(1, 11):
        img_file = path + str(i) + '/' + str(j) + '.pgm'
        out_file = path + str(i) + '/' + str(j + 20) + '.pgm'
        sp_noiseImg(img_file,  out_file,  0.02)

    for i in range(1, 41):
      for j in range(1, 11):
        img_file = path + str(i) + '/' + str(j) + '.pgm'
        out_file = path + str(i) + '/' + str(j + 30) + '.pgm'
        sp_noiseImg(img_file,  out_file,  0.005)

    for i in range(1, 41):
      for j in range(1, 11):
        img_file = path + str(i) + '/' + str(j) + '.pgm'
        out_file = path + str(i) + '/' + str(j + 40) + '.pgm'
        gasuss_noiseImg(img_file,  out_file)

def main():
    # noise()
    path = 'data/ATT_1/'
    # 旋转
    # for i in range(1, 41):
    #   for j in range(1, 51):
    #     img_file = path + str(i) + '/' + str(j) + '.pgm'
    #     out_file = path + str(i) + '/' + str(j + 50) + '.pgm'
    #     rotationImg(img_file,  out_file, 5)    

    # 缩放
    path_new = 'data/ATT_3/'
    for i in range(1, 41):
      for j in range(1, 11):
        img_file = path + str(i) + '/' + str(j) + '.pgm'
        out_file = path_new + str(i) + '/' + str(j) + '.pgm'
        resizeImg(img_file, out_file, 224, 224, 0) 
        # rotationImg(img_file,  out_file, -5)  
        # gasuss_noiseImg(img_file,  out_file)
        # print(img_file, ' ', out_file)   

if __name__ == '__main__':
    main()

    img_file = 'data/ATT_3/40/1.pgm'
    im = Image.open(img_file)
    im.show()