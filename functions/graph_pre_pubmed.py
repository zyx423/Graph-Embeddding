import os
import numpy as np
import sys
import pickle as pkl
import networkx as nx
import scipy.sparse as sp

path = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../..'))
path = path + '/Datasets/graph_datasets/'

def load_data(dataset):
    # load the data: x, tx, allx, graph
    names = ['allx', 'ally', 'graph', 'tx', 'x', 'ty', 'y']
    objects = []
    for i in range(len(names)):
        with open(path + 'pubmed/ind.{}.{}'.format(dataset, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
    allx, ally, graph, tx, x, ty, y = tuple(objects)
    test_idx_reorder = parse_index_file(path + 'pubmed/ind.{}.test.index'.format(dataset))
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

labels_inx = []

print("adj.shape is", adj.shape)
print("features.shape is", features.shape)
print("labels shape is", labels.shape)

for i in range(labels.shape[0]):
    row = labels.getrow(i)
    max_index = row.indices[row.data.argmax()] if row.nnz else 0
    labels_inx.append(max_index)

labels_inx = sp.coo_matrix(np.array(labels_inx))

sp.save_npz(path + '{}/features'.format(dataset), features)
sp.save_npz(path + '{}/labels'.format(dataset), labels_inx)
sp.save_npz(path + '{}/adjacency'.format(dataset), adj)

