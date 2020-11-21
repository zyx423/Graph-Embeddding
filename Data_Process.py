import scipy.io as scio
import torch
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.cluster import KMeans
import warnings
from sklearn.metrics import average_precision_score, roc_auc_score, adjusted_mutual_info_score

warnings.filterwarnings('ignore')
############################################ Load the datasets  ########################################################
class Load_Data():
    def __init__(self, dataset):
        self.dataset = dataset

    def CPU(self):
        # path1 = os.path.abspath('.')
        path = './Datasets/{}.mat'.format(self.dataset)
        data = scio.loadmat(path)

        Labels = data['Y']
        if Labels.shape[0] == 1:
            Labels = np.reshape(Labels, (Labels.shape[1], 1))
        features = data['X']

        if features.shape[1] == Labels.shape[0]:
           features = features.T  # change the data into n×d

        return features, Labels

    def GPU(self):
        path = './Dataset/{}.mat'.format(self.dataset)
        data = scio.loadmat(path)

        labels = data['Y']
        Labels = np.array(labels).flatten()

        if labels.shape[0] == 1:
            labels = np.reshape(labels, (labels.shape[1], 1))

        features = data['X']

        if features.shape[1] == labels.shape[0]:
            features = features.T

        return features, Labels

    def Graph(self):
        path = './Datasets/Graph_Datasets/{}/'.format(self.dataset)

        Features = sp.load_npz(path + 'Features.npz')
        Features = Features.toarray()

        Labels = sp.load_npz(path + 'Labels.npz')
        Labels = Labels.toarray()
        Labels = Labels.reshape(Labels.shape[1], 1)

        Adjacency = sp.load_npz(path + 'Adjacency.npz')
        Adjacency = Adjacency.toarray()

        return Features, Labels, Adjacency
################################ Graph construction for adjacency matrix and similarty matrix ###################
class Graph_Construction:
    # Input: n * d.
    def __init__(self, X):
        self.X = X

    def Middle(self):
            Inner_product = self.X.mm(self.X.T)
            Graph_middle = torch.sigmoid(Inner_product)
            return Graph_middle

    # Construct the adjacency matrix by KNN
    def KNN(self, k=9):
        n = self.X.shape[0]
        D = L2_distance_2(self.X, self.X)
        _, idx = torch.sort(D)
        S = torch.zeros(n, n)
        for i in range(n):
            id = torch.LongTensor(idx[i][1: (k + 1)])
            S[i][id] = 1
        S = (S + S.T) / 2
        return S

################################# Adjacency matrix normalization or pollution ##########################################
class Convolution_Kernel():
    def __init__(self, adjacency):
        self.adjacency = adjacency

    def Adjacency_Convolution(self):
        adj = self.adjacency + torch.eye(self.adjacency.shape[0])
        degrees = torch.Tensor(adj.sum(1))
        degrees_matrix_inv_sqrt = torch.diag(torch.pow(degrees, -0.5))
        return torch.mm(degrees_matrix_inv_sqrt, adj).mm(degrees_matrix_inv_sqrt)

    def Laplacian_Raw(self):
        degrees = torch.diag(torch.Tensor(self.adjacency.sum(1)).flatten())
        return degrees - self.adjacency

    ######## Laplacian matrix convolution
    def Laplacian_Convolution(self):
        S = self.adjacency + torch.eye(self.adjacency.size(0)) * 0.001
        degrees = (torch.Tensor(S.sum(1)).flatten())
        D = torch.diag(degrees)
        L = D - self.adjacency
        D_sqrt = torch.diag(torch.pow(degrees, -0.5))
        return D_sqrt.mm(L).mm(D_sqrt)


def L2_distance_2(A, B):
    A = A.T
    B = B.T
    AA = torch.sum(A*A, dim=0, keepdims=True)
    BB = torch.sum(B*B, dim=0, keepdims=True)
    AB = (A.T).mm(B)
    D = ((AA.T).repeat(1, BB.shape[1])) + (BB.repeat(AA.shape[1], 1)) - 2 * AB
    D = torch.abs(D)
    return D
######################################################### mySVM ########################################################
def mySVM(Latent_representation, Labels, scale=0.3):
    X_train, X_test, Y_train, Y_test = train_test_split(Latent_representation, Labels, test_size=scale, random_state=0)
    clf = svm.SVC(probability=True)
    clf.fit(X_train, Y_train)

    Pred_Y = clf.predict(X_test)
    score = f1_score(Pred_Y, Y_test, average='weighted')
    return score

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
        plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c, s = 5) # c=node_colors)
    plt.axis('off')
    # plt.legend()
    plt.gca.legend_ = None
    plt.show()

def get_weight_initial(d1, d2):
    bound = torch.sqrt(torch.Tensor([6.0 / (d1 + d2)]))
    nor_W = -bound + 2*bound*torch.rand(d1, d2)
    return torch.Tensor(nor_W)

########################################################## Pre Link Prediction #######################################
def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    degree_mat_inv_sqrt = sp.diags(np.power(np.array(adj_.sum(1)), -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt)
    return sparse_to_tuple(adj_normalized)

def construct_feed_dict(adj_normalized, adj, features, placeholders):
    # Construct feed dictionary
    feed_dict = dict()
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['adj']: adj_normalized})
    feed_dict.update({placeholders['adj_orig']: adj})
    return feed_dict

def mask_test_edges(adj, test_percent=10., val_percent=5.):
    """ Randomly removes some edges from original graph to create
    test and validation sets for link prediction task
    :param adj: complete sparse adjacency matrix of the graph
    :param test_percent: percentage of edges in test set
    :param val_percent: percentage of edges in validation set
    :return: train incomplete adjacency matrix, validation and test sets
    """
    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[None, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert adj.diagonal().sum() == 0

    edges_positive, _, _ = sparse_to_tuple(adj)
    # Filtering out edges from lower triangle of adjacency matrix
    edges_positive = edges_positive[edges_positive[:,1] > edges_positive[:,0],:]
    # val_edges, val_edges_false, test_edges, test_edges_false = None, None, None, None

    # number of positive (and negative) edges in test and val sets:
    num_test = int(np.floor(edges_positive.shape[0] / (100. / test_percent)))
    num_val = int(np.floor(edges_positive.shape[0] / (100. / val_percent)))

    # sample positive edges for test and val sets:
    edges_positive_idx = np.arange(edges_positive.shape[0])
    np.random.shuffle(edges_positive_idx)
    val_edge_idx = edges_positive_idx[:num_val]
    test_edge_idx = edges_positive_idx[num_val:(num_val + num_test)]
    test_edges = edges_positive[test_edge_idx] # positive test edges
    val_edges = edges_positive[val_edge_idx] # positive val edges
    train_edges = np.delete(edges_positive, np.hstack([test_edge_idx, val_edge_idx]), axis = 0) # positive train edges

    # the above strategy for sampling without replacement will not work for
    # sampling negative edges on large graphs, because the pool of negative
    # edges is much much larger due to sparsity, therefore we'll use
    # the following strategy:
    # 1. sample random linear indices from adjacency matrix WITH REPLACEMENT
    # (without replacement is super slow). sample more than we need so we'll
    # probably have enough after all the filtering steps.
    # 2. remove any edges that have already been added to the other edge lists
    # 3. convert to (i,j) coordinates
    # 4. swap i and j where i > j, to ensure they're upper triangle elements
    # 5. remove any duplicate elements if there are any
    # 6. remove any diagonal elements
    # 7. if we don't have enough edges, repeat this process until we get enough
    positive_idx, _, _ = sparse_to_tuple(adj) # [i,j] coord pairs for all true edges
    positive_idx = positive_idx[:,0]*adj.shape[0] + positive_idx[:,1] # linear indices
    test_edges_false = np.empty((0,2),dtype='int64')
    idx_test_edges_false = np.empty((0,),dtype='int64')

    while len(test_edges_false) < len(test_edges):
        # step 1:
        idx = np.random.choice(adj.shape[0]**2, 2*(num_test - len(test_edges_false)), replace = True)
        # step 2:
        idx = idx[~np.in1d(idx, positive_idx, assume_unique = True)]
        idx = idx[~np.in1d(idx, idx_test_edges_false, assume_unique = True)]
        # step 3:
        rowidx = idx // adj.shape[0]
        colidx = idx % adj.shape[0]
        coords = np.vstack((rowidx,colidx)).transpose()
        # step 4:
        lowertrimask = coords[:,0] > coords[:,1]
        coords[lowertrimask] = coords[lowertrimask][:,::-1]
        # step 5:
        coords = np.unique(coords, axis = 0) # note: coords are now sorted lexicographically
        np.random.shuffle(coords) # not anymore
        # step 6:
        coords = coords[coords[:,0] != coords[:,1]]
        # step 7:
        coords = coords[:min(num_test, len(idx))]
        test_edges_false = np.append(test_edges_false, coords, axis = 0)
        idx = idx[:min(num_test, len(idx))]
        idx_test_edges_false = np.append(idx_test_edges_false, idx)

    val_edges_false = np.empty((0,2), dtype = 'int64')
    idx_val_edges_false = np.empty((0,), dtype = 'int64')
    while len(val_edges_false) < len(val_edges):
        # step 1:
        idx = np.random.choice(adj.shape[0]**2, 2*(num_val - len(val_edges_false)), replace = True)
        # step 2:
        idx = idx[~np.in1d(idx, positive_idx, assume_unique = True)]
        idx = idx[~np.in1d(idx, idx_test_edges_false, assume_unique = True)]
        idx = idx[~np.in1d(idx, idx_val_edges_false, assume_unique = True)]
        # step 3:
        rowidx = idx // adj.shape[0]
        colidx = idx % adj.shape[0]
        coords = np.vstack((rowidx,colidx)).transpose()
        # step 4:
        lowertrimask = coords[:,0] > coords[:,1]
        coords[lowertrimask] = coords[lowertrimask][:,::-1]
        # step 5:
        coords = np.unique(coords, axis = 0) # note: coords are now sorted lexicographically
        np.random.shuffle(coords) # not any more
        # step 6:
        coords = coords[coords[:,0] != coords[:,1]]
        # step 7:
        coords = coords[:min(num_val, len(idx))]
        val_edges_false = np.append(val_edges_false, coords, axis = 0)
        idx = idx[:min(num_val, len(idx))]
        idx_val_edges_false = np.append(idx_val_edges_false, idx)

    # sanity checks:
    train_edges_linear = train_edges[:,0]*adj.shape[0] + train_edges[:,1]
    test_edges_linear = test_edges[:,0]*adj.shape[0] + test_edges[:,1]
    assert not np.any(np.in1d(idx_test_edges_false, positive_idx))
    assert not np.any(np.in1d(idx_val_edges_false, positive_idx))
    assert not np.any(np.in1d(val_edges[:,0]*adj.shape[0]+val_edges[:,1], train_edges_linear))
    assert not np.any(np.in1d(test_edges_linear, train_edges_linear))
    assert not np.any(np.in1d(val_edges[:,0]*adj.shape[0]+val_edges[:,1], test_edges_linear))

    # Re-build adj matrix
    data = np.ones(train_edges.shape[0])
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T
    return adj_train, val_edges, val_edges_false, test_edges, test_edges_false



def sigmoid(x):
    """ Sigmoid activation function
    :param x: scalar value
    :return: sigmoid activation
    """
    return 1 / (1 + np.exp(-x))

def get_roc_score(edges_pos, edges_neg, emb):
    """ Link Prediction: computes AUC ROC and AP scores from embeddings vectors,
    and from ground-truth lists of positive and negative node pairs
    :param edges_pos: list of positive node pairs
    :param edges_neg: list of negative node pairs
    :param emb: n*d matrix of embedding vectors for all graph nodes
    :return: Area Under ROC Curve (AUC ROC) and Average Precision (AP) scores
    """
    preds = []
    preds_neg = []
    for e in edges_pos:
        # Link Prediction on positive pairs
        preds.append(sigmoid(emb[e[0],:].dot(emb[e[1],:].T)))
    for e in edges_neg:
        # Link Prediction on negative pairs
        preds_neg.append(sigmoid(emb[e[0],:].dot(emb[e[1],:].T)))

    # Stack all predictions and labels
    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])

    # Computes metrics
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)
    return roc_score, ap_score

def clustering_latent_space(emb, label, nb_clusters=None):
    """ Node Clustering: computes Adjusted Mutual Information score from a
    K-Means clustering of nodes in latent embedding space
    :param emb: n*d matrix of embedding vectors for all graph nodes
    :param label: ground-truth node labels
    :param nb_clusters: int number of ground-truth communities in graph
    :return: Adjusted Mutual Information (AMI) score
    """
    if nb_clusters is None:
        nb_clusters = len(np.unique(label))
    # K-Means Clustering
    clustering_pred = KMeans(n_clusters = nb_clusters, init = 'k-means++').fit(emb).labels_
    # Compute metrics
    return adjusted_mutual_info_score(label, clustering_pred)