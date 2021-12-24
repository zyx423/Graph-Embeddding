import pandas as pd
import numpy as np
import scipy.io as scio
import scipy.sparse as sp

dataset = 'cora'
path = 'D:/OneDrive - mail.nwpu.edu.cn/Master/Optimal/Public/Datasets/Graph_Datasets/'
raw_data = pd.read_csv(path + '{}/{}.content'\
                       .format(dataset, dataset), sep = '\t', header = None, low_memory=False)
#####
num = raw_data.shape[0]
print('row_data.type = ', type(raw_data))

a = list(raw_data.index)
b = list(raw_data[0])
c = zip(b, a)
map = dict(c)

#
Features = raw_data.iloc[:, 1:-1]
Features = np.array(Features)
print('features.shape', Features.shape)

# labels
labels = pd.get_dummies(raw_data[raw_data.shape[1]-1])
True_Y = []

for row in range(labels.shape[0]): # df is the DataFrame
         for col in range(labels.shape[1]):
             if labels.iat[row, col] != 0:
                 True_Y.append(col + 1)

True_Y = np.array(True_Y)
print(True_Y.shape)

# adjacency matrix
raw_data_cites = pd.read_csv(path + '{}/{}.cites'.format(dataset, dataset), \
                             sep = '\t', header = None)


# 创建一个规模和邻接矩阵一样大小的矩阵
Adjacency = np.zeros((num, num))
# 跳过错误值
for i, j in zip(raw_data_cites[0], raw_data_cites[1]):
    try:
        x = map[i]
        y = map[j]
        Adjacency[x][y] = Adjacency[y][x] = 1 #有引用关系的样本点之间取1
    except:
        pass
    continue
print(Adjacency.shape)

########## It is recommended to use this method. It saves the matrix as the sparity matrix to save space in computer.###
# Features = sp.coo_matrix(Features)
# True_Y = sp.coo_matrix(True_Y)
# Adjacency = sp.coo_matrix(Adjacency)

# sp.save_npz(path + '{}/Features'.format(dataset), Features)
# sp.save_npz(path + '{}/Labels'.format(dataset), True_Y)
# sp.save_npz(path + '{}/Adjacency'.format(dataset), Adjacency)


##################### This way is to save the datasets as the text file. ###############################################
np.savetxt(path + '{}/features.txt'.format(dataset), Features)
np.savetxt(path + '{}/labels.txt'.format(dataset), True_Y)
np.savetxt(path + '{}/adjacency.txt'.format(dataset), Adjacency)
np.savetxt(path + '{}/raw_data_cites.txt'.format(dataset), np.array(raw_data_cites))

################### This way is to save the datasets as the data with which the MATLAB can deal. ######################
# scio.savemat('C:/Users/zyx423/OneDrive - mail.nwpu.edu.cn/Public/Datasets/{}.mat'.format(dataset), \
#              {'X': np.array(Features), 'Y': np.array(True_Y), 'adj': np.array(Adjacency)})




