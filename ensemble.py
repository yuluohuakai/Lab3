import pickle
import numpy as np
import math
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
class AdaBoostClassifier:
    '''A simple AdaBoost Classifier.'''

    def __init__(self, weak_classifier, n_weakers_limit):
        '''
        Initialize AdaBoostClassifier

        Args:
            weak_classifier: The class of weak classifier, which is recommend to be sklearn.tree.DecisionTreeClassifier.
            n_weakers_limit: The maximum number of weak classifier the model can use.
        '''
        self.weak_classifier = weak_classifier
        self.n_weakers_limit = n_weakers_limit
		

    def is_good_enough(self):
        '''Optional'''
       

    def fit(self,X,y):
        '''Build a boosted classifier from the training set (X, y).

        Args:
            X: An ndarray indicating the samples to be trained, which shape should be (n_samples,n_features).
            y: An ndarray indicating the ground-truth labels correspond to X, which shape should be (n_samples,1).
        '''
        self.alphaArray = []         #alphaArray记录每次迭代的alpha值
        self.classifierArray = []    #classifierArray记录每次迭代后的分类器
    
        W = 1 / X.shape[0] * np.ones((X.shape[0],))                #初始化W，使得每个样本的权重相同
        for m in range(self.n_weakers_limit):                      #迭代过程训练分类器
            classifier = DecisionTreeClassifier(max_depth = 3)
            classifier.fit(X, y,  sample_weight = W)
            error = 0
            y_predict = classifier.predict(X)
            result = y_predict + y
            for i in range(result.shape[0]):
                if(result[i] == 0):
                    error += W[i]
            if (error > 0.5):                                      #如果预测结果的准确率小于0.5，则break
                break
            alpha = 1/2 * np.log((1 - error) / error)
            Zm = 0
            for index in range(W.shape[0]):
                Zm = Zm + W[index] * math.exp(-alpha * y[index] * y_predict[index])
            for index in range(W.shape[0]):
                W[index] = W[index] * math.exp(-alpha * y[index] * y_predict[index]) / Zm      #更新W的值
            self.alphaArray.append(alpha)
            self.classifierArray.append(classifier) 
      
 
    def predict_scores(self, X, y):
        '''Calculate the weighted sum score of the whole base classifiers for given samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).

        Returns:
            An one-dimension ndarray indicating the scores of differnt samples, which shape should be (n_samples,1).
        '''
        y_predict = np.zeros((1,X.shape[0]))
        for i in range(len(self.alphaArray)):
            y_predict +=  self.alphaArray[i] * self.classifierArray[i].predict(X)
        y_pre = np.sign(y_predict)
        return y_pre[0]

            
        
        
        

    def predict(self, X, threshold=0):
        '''Predict the catagories for geven samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).
            threshold: The demarcation number of deviding the samples into two parts.

        Returns:
            An ndarray consists of predicted labels, which shape should be (n_samples,1).
        '''
        y_predict = self.predict_scores(X)
        for i in range(y_predict.shape[0]):
            if(y_predict[i] >= 0):
                y_predict[i] = 1                  #大于阈值标签设为1
            else:           
                y_predict[i] = -1                 #小于阈值标签设为-1
        return y_predict
                

       
           

    @staticmethod
    def save(model, filename):
        with open(filename, "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)