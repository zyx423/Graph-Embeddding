import pandas as pd
import numpy as np
import scipy.io as scio
import scipy.sparse as sp

dataset = 'cora'
raw_data = pd.read_csv('./{}/{}.content'.format(dataset, dataset), sep = '\t', header = None, low_memory=False)
#####
num = raw_data.shape[0]
print('row_data.type = ', type(raw_data))

a = list(raw_data.index)
b = list(raw_data[0])
c = zip(b, a)
map = dict(c)

#
features = raw_data.iloc[:, 1:-1]
features = np.array(features)
print('features.shape', features.shape)

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
raw_data_cites = pd.read_csv( './{}/{}.cites'.format(dataset, dataset), sep = '\t', header = None)


# creat the adjacency matrix
Adjacency = np.zeros((num, num))
# pass the error
for i, j in zip(raw_data_cites[0], raw_data_cites[1]):
    try:
        x = map[i]
        y = map[j]
        Adjacency[x][y] = Adjacency[y][x] = 1
    except:
        pass
    continue
print(Adjacency.shape)

## It is recommended to use this method. It saves the matrix as the sparity matrix to save space in computer.
features = sp.coo_matrix(features)
True_Y = sp.coo_matrix(True_Y)
Adjacency = sp.coo_matrix(Adjacency)

sp.save_npz('./{}/Features'.format(dataset), features)
sp.save_npz('./{}/Labels'.format(dataset), True_Y)
sp.save_npz('./{}/Adjacency'.format(dataset), Adjacency)


## This way is to save the datasets as the text file.
# np.savetxt(path + '{}/Features.txt'.format(dataset), Features)
# np.savetxt(path + '{}/Labels.txt'.format(dataset), True_Y)
# np.savetxt(path + '{}/Adjacency.txt'.format(dataset), Adjacency)

## This way is to save the datasets as the data with which the MATLAB can deal.
# scio.savemat('./{}.mat'.format(dataset), {'X': np.array(Features), 'Y': np.array(True_Y), 'adj': np.array(Adjacency)})




