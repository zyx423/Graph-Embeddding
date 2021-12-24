import sys
sys.path.append('../functions')
from model_GAE import *
from metrics import *
from sklearn.cluster import KMeans
import numpy as np
import torch
import time
import scipy.sparse as sp
import warnings
from link_prediction import *
warnings.filterwarnings('ignore')

######################################################### Setting #####################################################
DATASET = 'cora'
CLASSIFICATION = True
CLUSTERING = False
LINKRREDICTION = False
TSNE = False
path_result = "./embedding/VGAE/"
########################################## hyper-parameters##############################################################
epoch_num = 100
learning_rate = 5 * 1e-4

hidden_layer_1 = 1024
hidden_layer_2 = 128
################################### Load dataset   ######################################################################
if DATASET in ['cora', 'citeseer', 'pubmed']:
    load_data = LoadData(DATASET)
    features, labels, adjacency_matrix_raw = load_data.graph()

else:
    load_data = LoadData(DATASET)
    features, labels = load_data.mat()

################################### Calculate the adjacency matrix #######################################################
if ('adjacency_matrix_raw' in vars()) or (DATASET in ['cora', 'citeseer', 'pubmed']):
    print('adjacency matrix is raw')
    pass
else:
    print('adjacency matrix is caculated by KNN')
    graph = GraphConstruction(features)
    adjacency_matrix_raw = graph.knn()
################################### Link Prediction   ##################################################################
if LINKRREDICTION:
    adj_train, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(
        sp.coo_matrix(adjacency_matrix_raw))
    adjacency_matrix = adj_train.todense()
    adjacency_matrix = torch.Tensor(adjacency_matrix)
    features = torch.Tensor(features)
else:
    features = torch.Tensor(features)
    adjacency_matrix = torch.Tensor(adjacency_matrix_raw)

########################################### convolution_kernel ##############################################
convolution_kernel = ConvolutionKernel(adjacency_matrix)
adjacency_convolution_kernel = convolution_kernel.adjacency_convolution()
############################################ Results  Initialization ###################################################
acc_VGAE_total = []
nmi_VGAE_total = []
pur_VGAE_total = []

acc_VGAE_total_std = []
nmi_VGAE_total_std = []
pur_VGAE_total_std = []

F1_score = []

roc_score = []
ap_score = []
#################################################### Weight initialization  ###################################################
weight_mask = adjacency_matrix.view(-1) == 1
weight_tensor = torch.ones(weight_mask.size(0))
pos_weight = float(
    adjacency_matrix.shape[0] * adjacency_matrix.shape[0] - adjacency_matrix.sum()) / adjacency_matrix.sum()
weight_tensor[weight_mask] = 20


#######################################################  Loss Function #################################################
def loss_function(Graph_Reconstruction, Graph_Raw, H_2_mean, H_2_std):
    bce_loss = torch.nn.BCELoss(size_average=False, weight=weight_tensor)
    reconstruction_loss = bce_loss(Graph_Reconstruction.view(-1), Graph_Raw.view(-1))
    kl_divergence = -0.5 / adjacency_matrix.size(0) * (1 + 2 * H_2_std - H_2_mean ** 2 - torch.exp(H_2_std) ** 2).sum(
        1).mean()
    return reconstruction_loss, kl_divergence


############################################## Model ###################################################################
model_VGAE = myVGAE(features.shape[1], hidden_layer_1, hidden_layer_2)
optimzer = torch.optim.Adam(model_VGAE.parameters(), lr=learning_rate)
start_time = time.time()
#################################################### Train ###################################################
for epoch in range(epoch_num):
    latent_representation, graph_reconstruction, H_2_mean, H_2_std = model_VGAE(adjacency_convolution_kernel, features)
    reconstruction_loss, KL_divergence = loss_function(graph_reconstruction, adjacency_matrix, H_2_mean, H_2_std)
    loss = reconstruction_loss + KL_divergence

    optimzer.zero_grad()
    loss.backward()
    optimzer.step()

    latent_representation = latent_representation.cpu().detach().numpy()

    ################################################# save model ####################################################
    if epoch == 150:
        torch.save(model_VGAE.state_dict(), 'mode_VGAE_save.pt')
    ##################################################### Results  ####################################################
    if CLASSIFICATION and (epoch + 1) % 5 == 0:
        print("Epoch:{},Loss:{:.4f}".format(epoch + 1, loss.item()))
        score = my_SVM(latent_representation, labels, 0.3)
        print("Epoch[{}/{}], score = {}".format(epoch + 1, epoch_num, score))
        F1_score.append(score)

    elif CLUSTERING and (epoch + 1) % 5 == 0:
        print("Epoch[{}/{}], Reconstruction_Loss: {:.4f}, KL_Divergence: {:.4f}"
              .format(epoch + 1, epoch_num, reconstruction_loss.item(), KL_divergence.item()))

        acc_H2 = []
        nmi_H2 = []
        pur_H2 = []
        kmeans = KMeans(n_clusters=max(np.int_(labels).flatten()))
        for i in range(10):
            Y_pred_OK = kmeans.fit_predict(latent_representation)
            labels_K = np.array(labels).flatten()
            AM = clustering_metrics(Y_pred_OK, labels_K)
            acc, nmi, pur = AM.evaluationClusterModelFromLabel(print_msg=False)
            acc_H2.append(acc)
            nmi_H2.append(nmi)
            pur_H2.append(pur)

        print('ACC_VGAE=', 100 * np.mean(acc_H2), '\n', 'NMI_VGAE=', 100 * np.mean(nmi_H2), '\n', 'PUR_VGAE=',
              100 * np.mean(pur_H2))

        acc_VGAE_total.append(100 * np.mean(acc_H2))
        nmi_VGAE_total.append(100 * np.mean(nmi_H2))
        pur_VGAE_total.append(100 * np.mean(pur_H2))

        acc_VGAE_total_std.append(100 * np.std(acc_H2))
        nmi_VGAE_total_std.append(100 * np.std(nmi_H2))
        pur_VGAE_total_std.append(100 * np.std(pur_H2))
        #np.save(path_result + "{}.npy".format(epoch + 1), latent_representation)

    elif LINKRREDICTION and (epoch + 1) % 5 == 0:
        roc_score_temp, ap_score_temp = get_roc_score(test_edges, test_edges_false, latent_representation)
        roc_score.append(roc_score_temp)
        ap_score.append(ap_score_temp)

        print("Epoch: [{}]/[{}]".format(epoch + 1, epoch_num))
        print("AUC = {}".format(roc_score_temp))
        print("AP = {}".format(ap_score_temp))
        #np.save(path_result + "{}.npy".format(epoch + 1), latent_representation)
##################################################  Result #############################################################
if CLUSTERING:
    index_max = np.argmax(acc_VGAE_total)

    acc_VGAE_max = np.float(acc_VGAE_total[index_max])
    nmi_VGAE_max = np.float(nmi_VGAE_total[index_max])
    pur_VGAE_max = np.float(pur_VGAE_total[index_max])

    acc_STD = np.float(acc_VGAE_total_std[index_max])
    nmi_STD = np.float(nmi_VGAE_total_std[index_max])
    pur_std = np.float(pur_VGAE_total_std[index_max])

    print('ACC_VGAE_max={:.2f} +- {:.2f}'.format(acc_VGAE_max, acc_STD))
    print('NMI_VGAE_max={:.2f} +- {:.2f}'.format(nmi_VGAE_max, nmi_STD))
    print('PUR_VGAE_max={:.2f} +- {:.2f}'.format(pur_VGAE_max, pur_std))

elif CLASSIFICATION:
    index_max = np.argmax(F1_score)
    print("VGAE: F1-score_max is {:.2f}".format(100 * np.max(F1_score)))

elif LINKRREDICTION:
    print("VGAE: AUC_max is {:.2f}".format(100 * np.max(roc_score)))
    print("VGAE: AP_max is {:.2f}".format(100 * np.max(ap_score)))
########################################################### t- SNE #################################################
if TSNE:
    print("dataset is {}".format(DATASET))
    latent_representation_max = np.load(path_result + "{}.npy".format((index_max + 1) * 5))
    features = np.array(features)
    plot_embeddings(latent_representation_max, features, labels)
########################################################################################################################
end_time = time.time()
print("Running time is {}".format(end_time - start_time))
