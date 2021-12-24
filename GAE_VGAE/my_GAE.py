from model_GAE import *
from metrics import *
from data_process import *
from sklearn.cluster import KMeans
import numpy as np
import torch
import scipy.sparse as sp
import time
# from tensorboardX import SummaryWriter
import warnings
from link_prediction import *

warnings.filterwarnings('ignore')

#  features: X (n × d); adjacency: similarity matrix; labels: Y
#  Parameters that need to be entered manually
######################################################### Setting #####################################################
DATASET = 'att'
CLASSIFICATION = False
CLUSTERING = False
LINKPREDICTION =  True
TSNE = False
path_result = "./embedding/GAE/"
########################################## hyper-parameters##############################################################
epoch_num = 200
learning_rate = 1e-4

hidden_layer_1 = 1024
hidden_layer_2 = 128
############################################ Load dataset   #########################################################
if DATASET in ['cora', 'citeseer', 'pubmed']:
    load_data = LoadData(DATASET)
    features, labels, adjacency_matrix_raw = load_data.graph()

else:
    load_data = LoadData(DATASET)
    features, labels = load_data.mat()

################################### Calculate the adjacency matrix #########################################################
if ('adjacency_matrix_raw' in vars()) or (DATASET in ['cora', 'citeseer', 'pubmed']):
    print('adjacency matrix is raw')
    pass
else:
    print('adjacency matrix is caculated by KNN')
    graph = GraphConstruction(features)
    adjacency_matrix_raw = graph.knn()

################################### Link Prediction   ##################################################################
if LINKPREDICTION:
    adj_train, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(
        sp.coo_matrix(adjacency_matrix_raw))
    adjacency_matrix = adj_train.todense()
    adjacency_matrix = torch.Tensor(adjacency_matrix)
    features = torch.Tensor(features)
else:
    features = torch.Tensor(features)
    adjacency_matrix = torch.Tensor(adjacency_matrix_raw)


################################################ adjacency convolution ##################################################
convolution_kernel = ConvolutionKernel(adjacency_matrix)
adjacency_convolution_kernel = convolution_kernel.adjacency_convolution()

############################################ Results  Initialization ###################################################
ACC_GAE_total = []
NMI_GAE_total = []
PUR_GAE_total = []

ACC_GAE_total_STD = []
NMI_GAE_total_STD = []
PUR_GAE_total_STD = []

F1_score = []

roc_score = []
ap_score = []
#######################################  Model #########################################################################
# choose one from those two loss functions
mse_loss = torch.nn.MSELoss(size_average=False)
bce_loss = torch.nn.BCELoss(size_average=False)
model_GAE = myGAE(features.shape[1], hidden_layer_1, hidden_layer_2)
optimzer = torch.optim.Adam(model_GAE.parameters(), lr=learning_rate)

start_time = time.time()
#######################################  Train and result ################################################################
for epoch in range(epoch_num):
    graph_reconstruction, embedding = model_GAE(adjacency_convolution_kernel, features)
    loss = bce_loss(graph_reconstruction.view(-1), adjacency_matrix.view(-1))

    optimzer.zero_grad()
    loss.backward()
    optimzer.step()

    embedding = embedding.cpu().detach().numpy()

    ################################################# save model ####################################################
    if epoch == 150:
        torch.save(model_GAE.state_dict(), 'mode_GAE_save.pt')

    ##################################################### Results  ####################################################
    if CLASSIFICATION and (epoch + 1) % 5 == 0:
        print("Epoch:{},Loss:{:.4f}".format(epoch + 1, loss.item()))
        score = my_SVM(embedding, labels, 0.3)
        print("Epoch[{}/{}], score = {}".format(epoch + 1, epoch_num, score))
        F1_score.append(score)

    elif CLUSTERING and (epoch + 1) % 5 == 0:
        print("Epoch:{},Loss:{:.4f}".format(epoch + 1, loss.item()))
        ACC_H2 = []
        NMI_H2 = []
        PUR_H2 = []
        kmeans = KMeans(n_clusters=max(np.int_(labels).flatten()))
        for i in range(10):
            Y_pred_OK = kmeans.fit_predict(embedding)
            labels_K = np.array(labels).flatten()
            AM = clustering_metrics(Y_pred_OK, labels_K)
            ACC, NMI, PUR = AM.evaluationClusterModelFromLabel(print_msg=False)
            ACC_H2.append(ACC)
            NMI_H2.append(NMI)
            PUR_H2.append(PUR)

        ACC_GAE_total.append(100 * np.mean(ACC_H2))
        NMI_GAE_total.append(100 * np.mean(NMI_H2))
        PUR_GAE_total.append(100 * np.mean(PUR_H2))
        ACC_GAE_total_STD.append(100 * np.std(ACC_H2))
        NMI_GAE_total_STD.append(100 * np.std(NMI_H2))
        PUR_GAE_total_STD.append(100 * np.std(PUR_H2))

        print('ACC_H2=', 100 * np.mean(ACC_H2), '\n', 'NMI_H2=', 100 * np.mean(NMI_H2), '\n', 'PUR_H2=',
              100 * np.mean(PUR_H2))
        # np.save(path_result + "{}.npy".format(epoch + 1), embedding)

    elif LINKPREDICTION and (epoch + 1) % 5 == 0:
        roc_score_temp, ap_score_temp = get_roc_score(test_edges, test_edges_false, embedding)
        roc_score.append(roc_score_temp)
        ap_score.append(ap_score_temp)
        np.save(path_result + "{}.npy".format(epoch + 1), embedding)

        print("Epoch: [{}]/[{}]".format(epoch + 1, epoch_num))
        print("AUC = {}".format(roc_score_temp))
        print("AP = {}".format(ap_score_temp))

###############################################################Clustering  Result ##############################
if CLUSTERING:
    Index_MAX = np.argmax(ACC_GAE_total)

    print('ACC_GAE_max={:.2f} +- {:.2f}'.format(np.float(ACC_GAE_total[Index_MAX]), \
                                                np.float(ACC_GAE_total_STD[Index_MAX])))
    print('NMI_GAE_max={:.2f} +- {:.2f}'.format(np.float(NMI_GAE_total[Index_MAX]), \
                                                np.float(NMI_GAE_total_STD[Index_MAX])))
    print('PUR_GAE_max={:.2f} +- {:.2f}'.format(np.float(PUR_GAE_total[Index_MAX]), \
                                                np.float(PUR_GAE_total_STD[Index_MAX])))


elif CLASSIFICATION:
    Index_MAX = np.argmax(F1_score)
    print("GAE: F1-score_max is {:.2f}".format(100 * np.max(F1_score)))


elif LINKPREDICTION:
    print("GAE: AUC_max is {:.2f}".format(100 * np.max(roc_score)))
    print("GAE: AP_max is {:.2f}".format(100 * np.max(ap_score)))
########################################################### t- SNE #################################################
if TSNE:
    print("dataset is {}".format(DATASET))
    print("Index_Max = {}".format(Index_MAX))
    embedding_max = np.load(path_result + "{}.npy".format((Index_MAX + 1) * 5))
    plot_embeddings(embedding_max, np.array(features), labels)

########################################################################################################################
end_time = time.time()
print("Running time is {}".format(end_time - start_time))
