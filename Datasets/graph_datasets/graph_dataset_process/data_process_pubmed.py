import scipy.io as scio
import numpy as np
import sys
import pickle as pkl
import networkx as nx
import scipy.sparse as sp

def load_data(dataset):
    # load the data: x, tx, allx, graph
    names = ['allx', 'ally', 'graph', 'tx', 'x', 'ty', 'y']
    objects = []
    for i in range(len(names)):
        with open('./pubmed/ind.{}.{}'.format(dataset, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
    allx, ally, graph, tx, x, ty, y = tuple(objects)
    test_idx_reorder = parse_index_file('./pubmed/ind.{}.test.index'.format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = sp.coo_matrix(features)

    labels = sp.vstack((ally, ty)).tolil()
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    labels = sp.coo_matrix(labels)

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = sp.coo_matrix(adj)

    return adj, features, labels

def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

dataset = 'pubmed'
adj, features, labels = load_data(dataset)

print("adj.shape is", adj.shape)
print("features.shape is", features.shape)
print("labels shape is", labels.shape)


## It is recommended to use this method. It saves the matrix as the sparity matrix to save space in computer.
sp.save_npz('./{}/Features'.format(dataset), features)
sp.save_npz('./{}/Labels'.format(dataset), labels)
sp.save_npz('./{}/Adjacency'.format(dataset), adj)

