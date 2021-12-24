from data_process import *
from sklearn.cluster import KMeans
import numpy as np
import scipy.sparse as sp
import warnings
from sklearn.metrics import average_precision_score, roc_auc_score, adjusted_mutual_info_score

warnings.filterwarnings('ignore')


#  features: X (n × d); adjacency: similarity matrix; labels: Y
#  Parameters that need to be entered manual


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
    edges_positive = edges_positive[edges_positive[:, 1] > edges_positive[:, 0], :]
    # val_edges, val_edges_false, test_edges, test_edges_false = None, None, None, None

    # number of positive (and negative) edges in test and val sets:
    num_test = int(np.floor(edges_positive.shape[0] / (100. / test_percent)))
    num_val = int(np.floor(edges_positive.shape[0] / (100. / val_percent)))

    # sample positive edges for test and val sets:
    edges_positive_idx = np.arange(edges_positive.shape[0])
    np.random.shuffle(edges_positive_idx)
    val_edge_idx = edges_positive_idx[:num_val]
    test_edge_idx = edges_positive_idx[num_val:(num_val + num_test)]
    test_edges = edges_positive[test_edge_idx]  # positive test edges
    val_edges = edges_positive[val_edge_idx]  # positive val edges
    train_edges = np.delete(edges_positive, np.hstack([test_edge_idx, val_edge_idx]), axis=0)  # positive train edges

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
    positive_idx, _, _ = sparse_to_tuple(adj)  # [i,j] coord pairs for all true edges
    positive_idx = positive_idx[:, 0] * adj.shape[0] + positive_idx[:, 1]  # linear indices
    test_edges_false = np.empty((0, 2), dtype='int64')
    idx_test_edges_false = np.empty((0,), dtype='int64')

    while len(test_edges_false) < len(test_edges):
        # step 1:
        idx = np.random.choice(adj.shape[0] ** 2, 2 * (num_test - len(test_edges_false)), replace=True)
        # step 2:
        idx = idx[~np.in1d(idx, positive_idx, assume_unique=True)]
        idx = idx[~np.in1d(idx, idx_test_edges_false, assume_unique=True)]
        # step 3:
        rowidx = idx // adj.shape[0]
        colidx = idx % adj.shape[0]
        coords = np.vstack((rowidx, colidx)).transpose()
        # step 4:
        lowertrimask = coords[:, 0] > coords[:, 1]
        coords[lowertrimask] = coords[lowertrimask][:, ::-1]
        # step 5:
        coords = np.unique(coords, axis=0)  # note: coords are now sorted lexicographically
        np.random.shuffle(coords)  # not anymore
        # step 6:
        coords = coords[coords[:, 0] != coords[:, 1]]
        # step 7:
        coords = coords[:min(num_test, len(idx))]
        test_edges_false = np.append(test_edges_false, coords, axis=0)
        idx = idx[:min(num_test, len(idx))]
        idx_test_edges_false = np.append(idx_test_edges_false, idx)

    val_edges_false = np.empty((0, 2), dtype='int64')
    idx_val_edges_false = np.empty((0,), dtype='int64')
    while len(val_edges_false) < len(val_edges):
        # step 1:
        idx = np.random.choice(adj.shape[0] ** 2, 2 * (num_val - len(val_edges_false)), replace=True)
        # step 2:
        idx = idx[~np.in1d(idx, positive_idx, assume_unique=True)]
        idx = idx[~np.in1d(idx, idx_test_edges_false, assume_unique=True)]
        idx = idx[~np.in1d(idx, idx_val_edges_false, assume_unique=True)]
        # step 3:
        rowidx = idx // adj.shape[0]
        colidx = idx % adj.shape[0]
        coords = np.vstack((rowidx, colidx)).transpose()
        # step 4:
        lowertrimask = coords[:, 0] > coords[:, 1]
        coords[lowertrimask] = coords[lowertrimask][:, ::-1]
        # step 5:
        coords = np.unique(coords, axis=0)  # note: coords are now sorted lexicographically
        np.random.shuffle(coords)  # not any more
        # step 6:
        coords = coords[coords[:, 0] != coords[:, 1]]
        # step 7:
        coords = coords[:min(num_val, len(idx))]
        val_edges_false = np.append(val_edges_false, coords, axis=0)
        idx = idx[:min(num_val, len(idx))]
        idx_val_edges_false = np.append(idx_val_edges_false, idx)

    # sanity checks:
    train_edges_linear = train_edges[:, 0] * adj.shape[0] + train_edges[:, 1]
    test_edges_linear = test_edges[:, 0] * adj.shape[0] + test_edges[:, 1]
    assert not np.any(np.in1d(idx_test_edges_false, positive_idx))
    assert not np.any(np.in1d(idx_val_edges_false, positive_idx))
    assert not np.any(np.in1d(val_edges[:, 0] * adj.shape[0] + val_edges[:, 1], train_edges_linear))
    assert not np.any(np.in1d(test_edges_linear, train_edges_linear))
    assert not np.any(np.in1d(val_edges[:, 0] * adj.shape[0] + val_edges[:, 1], test_edges_linear))

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
        preds.append(sigmoid(emb[e[0], :].dot(emb[e[1], :].T)))
    for e in edges_neg:
        # Link Prediction on negative pairs
        preds_neg.append(sigmoid(emb[e[0], :].dot(emb[e[1], :].T)))

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
    clustering_pred = KMeans(n_clusters=nb_clusters, init='k-means++').fit(emb).labels_
    # Compute metrics
    return adjusted_mutual_info_score(label, clustering_pred)
