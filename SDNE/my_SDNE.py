from sklearn.datasets import make_moons
import sys
from model_SDNE import *
from metrics import *
from sklearn.cluster import KMeans
import numpy as np
import torch

import warnings
warnings.filterwarnings('ignore')


#  features: X (n × d);
#  Adjacency: N*N;
#  Labels: Y
#  Parameters that need to be entered manually

######################################################### Setting #####################################################
DATASET = 'cora'
CLASSIFICATION = True
CLUSTERING = False
TSNE = True
path_result = "./embedding/"
########################################## hyper-parameters##############################################################
epoch_num = 200
learning_rate = 5 * 1e-4

hidden_layer_1 = 1024
hidden_layer_2 = 128
################################### Load dataset   ######################################################################
if (DATASET is "cora") or (DATASET is "citeseer"):
    load_data = LoadData(DATASET)
    features, labels, adjacency_matrix = load_data.graph()
    features = torch.Tensor(features)
    adjacency_matrix = torch.Tensor(adjacency_matrix)
else:
    load_data = LoadData(DATASET)
    features, labels = load_data.mat()
    features = torch.Tensor(features)
################################### Calculate the adjacency matrix #########################################################
if('adjacency_matrix' in vars()):
    print('Adjacency matrix is raw')
    pass
else:
    print('adjacency matrix is caculated by KNN')
    graph = GraphConstruction(features.numpy())
    adjacency_matrix = graph.knn()
################################################ adjacency convolution ##################################################
convolution_kernel = ConvolutionKernel(adjacency_matrix)
laplacian_convolution_kernel = convolution_kernel.laplacian_convolution()
########################################## hyper-parameters##############################################################
epoch_num = 40
learning_rate = 1e-4
Lambda = 1

input_dim = adjacency_matrix.shape[0]
hidden_layer_1 = 1024
hidden_layer_2 = 128

B = adjacency_matrix * (20 - 1) + 1
############################################ Results  Initialization ###################################################
acc_SDNE_total = []
nmi_SDNE_total = []
pur_SDNE_total = []

acc_SDNE_total_std = []
nmi_SDNE_total_std = []
pur_SDNE_total_std = []

F1_score = []

#######################################  Model #########################################################################
mse_loss = torch.nn.MSELoss(size_average=False)
model_SDNE = mySDNE(input_dim, hidden_layer_1, hidden_layer_2)
optimzer = torch.optim.Adam(model_SDNE.parameters(), lr=learning_rate)
#######################################  Train and result ################################################################
for epoch in range(epoch_num):
    embedding, graph_reconstrction = model_SDNE(adjacency_matrix)
    loss_1st = torch.norm((graph_reconstrction - adjacency_matrix) * B, p ='fro')
    loss_2st = torch.trace((embedding.T).mm(laplacian_convolution_kernel).mm(embedding))
    loss = loss_1st + Lambda * loss_2st

    optimzer.zero_grad()
    loss.backward()
    optimzer.step()

    embedding = embedding.cpu().detach().numpy()
    ##################################################### Results  ####################################################
    if CLASSIFICATION and (epoch + 1) % 5 == 0:
        print("Epoch:{},Loss:{:.4f}".format(epoch + 1, loss.item()))
        score = my_SVM(embedding, labels, 0.3)
        print("Epoch[{}/{}], score = {}".format(epoch + 1, epoch_num, score))
        F1_score.append(score)
        np.save(path_result + "{}.npy".format(epoch + 1), embedding)

    elif CLUSTERING and (epoch + 1) % 5 == 0:
        print("Epoch:{},Loss:{:.4f}".format(epoch + 1, loss.item()))

        ACC_H2 = []
        NMI_H2 = []
        PUR_H2 = []
        ##############################################################################
        kmeans = KMeans(n_clusters=max(np.int_(labels).flatten()))
        for i in range(10):
            Y_pred_OK = kmeans.fit_predict(embedding)
            Y_pred_OK = np.array(Y_pred_OK)
            labels = np.array(labels).flatten()
            AM = clustering_metrics(Y_pred_OK, labels)
            ACC, NMI, PUR = AM.evaluationClusterModelFromLabel(print_msg=False)
            ACC_H2.append(ACC)
            NMI_H2.append(NMI)
            PUR_H2.append(PUR)
        print('ACC_H2=', 100 * np.mean(ACC_H2), '\n', 'NMI_H2=', 100 * np.mean(NMI_H2), '\n', 'PUR_H2=',
              100 * np.mean(PUR_H2))
        acc_SDNE_total.append(100 * np.mean(ACC_H2))
        nmi_SDNE_total.append(100 * np.mean(NMI_H2))
        pur_SDNE_total.append(100 * np.mean(PUR_H2))

        acc_SDNE_total_std.append(100 * np.std(ACC_H2))
        nmi_SDNE_total_std.append(100 * np.std(NMI_H2))
        pur_SDNE_total_std.append(100 * np.std(PUR_H2))
        np.save(path_result + "{}.npy".format(epoch + 1), embedding)
##################################################  Result #############################################################
if CLUSTERING:

    index_max = np.argmax(acc_SDNE_total)

    acc_SDNE_max = np.float(acc_SDNE_total[index_max])
    nmi_SDNE_max = np.float(nmi_SDNE_total[index_max])
    pur_SDNE_max = np.float(pur_SDNE_total[index_max])

    acc_std = np.float(acc_SDNE_total_std[index_max])
    nmi_std = np.float(nmi_SDNE_total_std[index_max])
    pur_std = np.float(pur_SDNE_total_std[index_max])

    print('ACC_SDNE_max={:.2f} +- {:.2f}'.format(acc_SDNE_max, acc_std))
    print('NMI_SDNE_max={:.2f} +- {:.2f}'.format(nmi_SDNE_max, nmi_std))
    print('PUR_SDNE_max={:.2f} +- {:.2f}'.format(pur_SDNE_max, pur_std))

elif CLASSIFICATION:
    index_max = np.argmax(F1_score)
    print("SDNE: F1-score_max is {:.2f}".format(100 * np.max(F1_score)))

########################################################### t- SNE #################################################
if TSNE:
    print("dataset is {}".format(DATASET))
    latent_representation_max = np.load(path_result + "{}.npy".format((index_max + 1) * 5))
    features = np.array(features)
    plot_embeddings(latent_representation_max, features, labels)