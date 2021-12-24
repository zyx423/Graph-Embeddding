import scipy.io as scio
import torch
import random
import numpy as np
import scipy.sparse as sp
from DataProcess import *
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

path_result = "D:/OneDrive - mail.nwpu.edu.cn/Optimal/BAGE/BAGE_Code/Latent_representation/"

Index_Max = 7
Dataset = 'COIL20'
load_data = Load_Data(Dataset)
Features, Labels = load_data.CPU()

Latent_representation = np.load(path_result + "{}.npy".format((Index_Max + 1) * 5))
Labels = np.array(Labels).flatten()
#### scale is the test dataset
scale = 0.3

X_train, X_test, Y_train, Y_test = train_test_split(Latent_representation, Labels, test_size=scale, random_state=0)

clf = svm.SVC(probability=True)
clf.fit(X_train, Y_train)

Pred_Y = clf.predict(X_test)
score = f1_score(Pred_Y, Y_test, average='weighted')
print(score)
