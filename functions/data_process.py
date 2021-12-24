import scipy.io as scio
import torch
import random
import numpy as np
import scipy.sparse as sp
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


############################################ Load the datasets  ########################################################
class LoadData():
    def __init__(self, dataset):
        self.dataset = dataset
        self.path = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../'))
    def mat(self):

        path = self.path + '/Datasets/{}.mat'.format(self.dataset)
        data = scio.loadmat(path)

        labels = data['Y']
        if labels.shape[0] == 1:
            labels = np.reshape(labels, (labels.shape[1], 1))
        features = data['X']

        if features.shape[1] == labels.shape[0]:
            features = features.T  # change the data into n×d

        return features, labels


    def graph(self):

        path = self.path + '/Datasets/graph_datasets/{}/'.format(self.dataset)

        features = sp.load_npz(path + 'features.npz')
        features = features.toarray()

        labels = sp.load_npz(path + 'labels.npz')
        labels = labels.toarray()
        labels = labels.reshape(labels.shape[1], 1)

        adjacency = sp.load_npz(path + 'adjacency.npz')
        adjacency = adjacency.toarray()

        return features, labels, adjacency


######################################### Data preprocess ##############################################################
class Normalized():
    def __init__(self, X):
        self.X = X

    def Normal(self):
        # X: n×d，normalize each dimension of X
        return (self.X - np.mean(self.X, axis=0)) / (np.std(self.X, axis=0, ddof=1) + 1e-4)

    def Length(self):
        # X:n*d, Make each row of X equal to each other.
        meth = np.sum(self.X, axis=1)
        meth = meth.reshape(meth.shape[0], 1)
        return self.X / (meth + 1e-6)

    def MinMax(self):
        # Normalize matrix by Min and Max
        # X: n*d, apply to each columns
        min = np.min(self.X, axis=0)
        max = np.max(self.X, axis=0)
        return (self.X - min) / (max - min + 1e-6)


################################ Graph construction for adjacency matrix and similarty matrix ###################
class GraphConstruction():
    # Input: n * d.
    def __init__(self, X):
        self.X = X

    def middle(self):
        Inner_product = self.X.mm(self.X.T)
        Graph_middle = torch.sigmoid(Inner_product)
        return Graph_middle

    # Construct the adjacency matrix by KNN
    def knn(self, k=9, issymmetry=False):
        n = self.X.shape[0]
        D = l2_distance(self.X, self.X)
        idx = np.argsort(D, axis=1)
        S = np.zeros((n, n))
        for i in range(n):
            id = idx[i][1: (k + 1)]
            S[i][id] = 1
        if issymmetry:
            S = (S + S.T) / 2
        return S

    # Construct the adjacency matrix by CAN
    def can(self, k=9, issymmetry=True):
        n = self.X.shape[0]
        D = l2_distance(self.X, self.X)
        idx = np.argsort(D, axis=1)
        S = np.zeros((n, n))
        for i in range(n):
            id = idx[i][1: (k + 1 + 1)]
            di = D[i][id]
            S[i][id] = ((di[k].repeat(di.shape[0])) - di) / (k * di[k] - np.sum(di[0:k]) + 1e-4)
        if issymmetry:
            S = (S + S.T) / 2
        return S

    def adjacency_incomplete(self, scale):
        Adjacency = np.array(self.adjacency)
        raw, col = np.nonzero(Adjacency)
        Num_nozero = len(raw)
        Num_setzero = round(scale * Num_nozero)

        Index_setzero = random.sample(range(0, Num_nozero), Num_setzero)

        for i in range(Num_setzero):
            raw_0 = raw[Index_setzero[i]]
            col_0 = col[Index_setzero[i]]
            Adjacency[raw_0][col_0] = 0

        left_0, _ = np.nonzero(Adjacency)
        print("all nozero elements is {}, setzero is {}, leftnozero is {}".format(Num_nozero, len(Index_setzero),
                                                                                  len(left_0)))
        return torch.Tensor(Adjacency)


################################# Adjacency matrix normalization or pollution ###########################################
class ConvolutionKernel():
    def __init__(self, adjacency):
        self.adjacency = adjacency

    def adjacency_convolution(self):
        adj = self.adjacency + torch.eye(self.adjacency.shape[0])
        degrees = torch.Tensor(adj.sum(1))
        degrees_matrix_inv_sqrt = torch.diag(torch.pow(degrees, -0.5))
        return torch.mm(degrees_matrix_inv_sqrt, adj).mm(degrees_matrix_inv_sqrt)

    def laplacian_raw(self):
        degrees = torch.diag(torch.Tensor(self.adjacency.sum(1)).flatten())
        return degrees - self.adjacency

    ######## Laplacian matrix convolution
    def laplacian_convolution(self):
        S = self.adjacency + torch.eye(self.adjacency.size(0)) * 0.001
        degrees = (torch.Tensor(S.sum(1)).flatten())
        D = torch.diag(degrees)
        L = D - self.adjacency
        D_sqrt = torch.diag(torch.pow(degrees, -0.5))
        return D_sqrt.mm(L).mm(D_sqrt)

##########################################  Some functions ###############################################################

################# Make the imcomplement for the adjacency matrix #############################
def adjacency_incomplete(adjacency, scale):
    adjacency = np.array(adjacency)
    raw, col = np.nonzero(adjacency)
    Num_nozero = len(raw)
    Num_setzero = round(scale * Num_nozero)

    Index_setzero = random.sample(range(0, Num_nozero), Num_setzero)

    for i in range(Num_setzero):
        raw_0 = raw[Index_setzero[i]]
        col_0 = col[Index_setzero[i]]
        adjacency[raw_0][col_0] = 0

    left_0, _ = np.nonzero(adjacency)
    print("all nozero elements is {}, setzero is {}, leftnozero is {}".format(Num_nozero, len(Index_setzero),
                                                                              len(left_0)))
    return torch.Tensor(adjacency)


## The initialization of the weighted matrix
def get_weight_initial(d1, d2):
    bound = torch.sqrt(torch.Tensor([6.0 / (d1 + d2)]))
    nor_W = -bound + 2 * bound * torch.rand(d1, d2)
    return torch.Tensor(nor_W)


# L_2 norm squared between matrix samples
# each column is a sample

def l2_distance(A, B):
    AA = np.sum(A * A, axis=1, keepdims=True)
    BB = np.sum(B * B, axis=1, keepdims=True)
    AB = (A).dot(B.T)
    D = (AA.repeat(BB.shape[0], axis=1)) + ((BB.T).repeat(AA.shape[0], axis=0)) - 2 * AB
    D = np.abs(D)
    return D


############################################# plot the t-SNE ###########################################################
def plot_embeddings(embeddings, Features, Labels):
    # norm = Normalized(embeddings)
    # embeddings = norm.MinMax()

    emb_list = []
    for k in range(Features.shape[0]):
        emb_list.append(embeddings[k])
    emb_list = np.array(emb_list)

    model = TSNE(n_components=2, init="pca")
    # model = TSNE(n_components=2)
    node_pos = model.fit_transform(emb_list)

    color_idx = {}
    for i in range(Features.shape[0]):
        color_idx.setdefault(Labels[i][0], [])
        color_idx[Labels[i][0]].append(i)

    for c, idx in color_idx.items():
        plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c, s=5)  # c=node_colors)
    plt.axis('off')
    # plt.legend()
    plt.gca.legend_ = None
    plt.show()


######################################################### mySVM ########################################################
def my_SVM(embedding, labels, label_ratio =0.3):
    X_train, X_test, Y_train, Y_test = train_test_split(embedding, labels, test_size=label_ratio, random_state=0)
    clf = svm.SVC(probability=True)
    clf.fit(X_train, Y_train)

    Pred_Y = clf.predict(X_test)
    score = f1_score(Pred_Y, Y_test, average='weighted')
    return score
