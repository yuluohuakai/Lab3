#实验日期:2017/12/16
#实验目的：理解Adaboost的原理，利用Adaboost解决人脸分类问题
#实验输入：500张含有人脸的图片，500张不含人脸的图片
#实验输出：分类正确率

import numpy as np
import math
import random
import matplotlib.pyplot as plt
import datetime
import urllib, PIL
from PIL import Image
from feature import *
from ensemble import *
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

def load_image(fileDir):                              #此函数读取图片并将图片变成大小为24*24的灰度图
    datas = []
    # 读取图片
    imgs = os.listdir(fileDir)
    imgNum = len(imgs)
    for i in range (imgNum):
        im = Image.open(fileDir + "/"+imgs[i]) 
        #width,height = im.size
        im = im.resize((24, 24),Image.ANTIALIAS)     #改变图片的大小为24*24
        im = im.convert("L")                         #将图片转为灰度图
        data = im.getdata()
        data = np.array(data)
        datas.append(data)
        
    return datas

if __name__ == "__main__":
   
   faceFileDir = "E:/datasets/original/face"
   nonfaceFileDir = "E:/datasets/original/nonface"

   data_positive = np.array(load_image(faceFileDir))
   data_negative = np.array(load_image(nonfaceFileDir))


   data = np.concatenate([data_positive, data_negative],axis=0)


   featureList = []
   for i in range(data.shape[0]):
       featureProcess = NPDFeature(data[i])                   #利用NPDFeature提取特征并保存
       NPD_feature = featureProcess.extract()
       featureList.append(NPD_feature)

   X = np.array(featureList)
   
   output = open('E:/datasets/original/data.pkl', 'wb')       
   pickle.dump(X, output)                                     #利用dump函数来将特征保存在缓存
   output.close()
   

   in_put = open('E:/datasets/original/data.pkl', 'rb')
   X = pickle.load(in_put)                                    #利用load函数来读取特征
   in_put.close()
   

   y_positive = np.ones((500,))                               #设定正类（包含人脸图片）标签为1
   y_negative = -np.ones((500,))                              #设定负类（不包含人脸图片）标签为-1
   y = np.concatenate([y_positive, y_negative],axis=0)

   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)   #随机切分训练集和测试集，切分比例7:3

   weak_classifier = DecisionTreeClassifier(max_depth = 3)                   #设置决策树最大深度3
   n_weakers_limit = 8                                                       #设置分类器个数为8
   adaboost = AdaBoostClassifier(weak_classifier, n_weakers_limit)
   adaboost.fit(X_train, y_train)
   y_predict = adaboost.predict_scores(X_test,y_test)

        
   class_name = ['face','nonface']
   writeReport = classification_report(y_test, y_predict, target_names=class_name)       
   reportFile = open('E:/datasets/original/report.txt', 'w')         
   reportFile.write(writeReport)                                            #写入预测结果
   reportFile.close()