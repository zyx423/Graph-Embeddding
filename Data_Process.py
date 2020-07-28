import scipy.io as scio
import torch
import os
import numpy as np
import sys
import pickle as pkl
import scipy.sparse as sp

############################################ Load the datasets  ########################################################

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

    # Construct the adjacency matrix by CAN
    def CAN(self, k):
        # Input: n * d.
        n = self.X.shape[0]
        D = L2_distance_2(self.X, self.X)
        _, idx = torch.sort(D)
        S = torch.zeros(n, n)
        for i in range(n):
            id = torch.LongTensor(idx[i][1:k + 1 + 1])
            di = D[i][id]
            S[i][id] = (torch.Tensor(di[k].repeat(di.shape[0])) - di) / (k * di[k] - torch.sum(di[0:k]) + 1e-4)
        S = (S + S.T) / 2
        return S

    # Reconstruct adjacency matrix by Rui Zhang
    def Reconstruct(self, beta):
        # X : n*d
        n, d = self.X.size()
        A = (self.X).mm(self.X.T)
        A[A < 1e-4] = 0
        F = torch.sigmoid(A)
        S = torch.zeros(n, n)
        E = L2_distance_2(self.X, self.X)
        A_alpha = (F - beta * E)
        for i2 in range(n):
            tran = EProjSimplex_new(A_alpha[:, i2:i2 + 1], 1)
            S[:, i2:i2 + 1] = tran
        S = (S + S.T) / 2
        return S


############################################## Laplacian matrix #########################################################


##########################################  Some fuctions ###############################################################
# The construction of adjacency matrix， X：n×d


# Construct the adjacency matrix by inner and add the sigmoid
## The initialization of the weighted matrix
