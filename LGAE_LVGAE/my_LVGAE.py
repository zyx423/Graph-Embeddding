from model_LGAE import *
from metrics import *
import scipy.io as scio
from sklearn.cluster import KMeans
import numpy as np
import torch
import time
from link_prediction import *
import warnings
warnings.filterwarnings('ignore')
######################################################### Setting #####################################################
DATASET = "att"
CLASSIFICATION = False
CLUSTERING = False
LINKPREDICTION = True
TSNE = False
path_result = "./embedding/LVAE/"
########################################## hyper-parameters##############################################################
epoch_num = 150
learning_rate = 1e-4
hidden_layer_1 = 128

################################### Load dataset   ######################################################################
if ((DATASET == "cora") or (DATASET == "citeseer")):
    load_data = LoadData(DATASET)
    features, labels, adjacency_matrix_raw = load_data.garph()

else:
    load_data = LoadData(DATASET)
    features, labels = load_data.mat()

################################### Calculate the adjacency matrix #########################################################
if (('adjacency_matrix_raw' in vars()) or (DATASET == "cora") or (DATASET == "citeseer")):
    print('adjacency matrix is raw')
    pass
else:
    print('adjacency matrix is caculated by CAN')
    graph = GraphConstruction(features)
    adjacency_matrix_raw = graph.knn()

################################### Link Prediction   ##################################################################
if LINKPREDICTION:
    adj_train, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(sp.coo_matrix(adjacency_matrix_raw))
    adjacency_matrix = adj_train.todense()
    adjacency_matrix = torch.Tensor(adjacency_matrix)
    features = torch.Tensor(features)
else:
    features = torch.Tensor(features)
    adjacency_matrix = torch.Tensor(adjacency_matrix_raw)
################################################## convolution_kernel ##############################################
convolution_kernel = ConvolutionKernel(adjacency_matrix)
adjacency_convolution_kernel = convolution_kernel.adjacency_convolution()


############################################ Results  Initialization ###################################################
acc_LVGAE_total = []
nmi_LVGAE_total = []
pur_LVGAE_total = []

acc_LVGAE_total_std = []
nmi_LVGAE_total_std = []
pur_LVGAE_total_std = []

F1_score = []

roc_score = []
ap_score = []
#################################################### Weight initialization  ###################################################
weight_mask = adjacency_matrix.view(-1) == 1
weight_tensor = torch.ones(weight_mask.size(0))
pos_weight = float(adjacency_matrix.shape[0] * adjacency_matrix.shape[0] - adjacency_matrix.sum()) / adjacency_matrix.sum()
weight_tensor[weight_mask] = 100

#######################################################  Loss Function #################################################
def loss_function(Graph_Reconstruction, Graph_Raw, H_2_mean, H_2_std):
    # mse_loss = torch.nn.MSELoss(size_average=False)
    # Reconstruction_Loss = mse_loss(Graph_Reconstruction, Graph_Raw)
    bce_loss = torch.nn.BCELoss(size_average=False, weight=weight_tensor)
    reconstruction_loss = bce_loss(Graph_Reconstruction.view(-1), Graph_Raw.view(-1))
    kl_divergence = -0.5 / Graph_Reconstruction.size(0) * (1 + 2 * H_2_std - H_2_mean ** 2 - torch.exp(H_2_std) ** 2).sum(1).mean()
    return reconstruction_loss, kl_divergence

############################################## Model ###################################################################
model_LVGAE = myLVGAE(features.shape[1], hidden_layer_1)
optimzer = torch.optim.Adam(model_LVGAE.parameters(), lr=learning_rate)
start_time = time.time()
#################################################### Train ###################################################
for epoch in range(epoch_num):
    embedding, graph_reconstruction, H_1_mean, H_1_std = model_LVGAE(adjacency_convolution_kernel, features)
    reconstruction_loss, kl_divergence = loss_function(graph_reconstruction, adjacency_matrix, H_1_mean, H_1_std)
    loss = reconstruction_loss + kl_divergence

    optimzer.zero_grad()
    loss.backward()
    optimzer.step()

    embedding = embedding.cpu().detach().numpy()
    ################################################# save model ####################################################
    if epoch == 150:
        torch.save(model_LVGAE.state_dict(), 'mode_LVGAE_save.pt')
    ##################################################### Results  ####################################################
    if CLASSIFICATION and (epoch + 1) % 5 == 0:
        print("Epoch:{},Loss:{:.4f}".format(epoch + 1, loss.item()))
        score = my_SVM(embedding, labels, 0.3)
        print("Epoch[{}/{}], score = {}".format(epoch + 1, epoch_num, 100 * score))
        F1_score.append(score)

    elif CLUSTERING and (epoch + 1) % 5 == 0:
        print("Loss = {}".format(loss.item()))
        print("Epoch[{}/{}], Reconstruction_Loss: {:.4f}, KL_Divergence: {:.4f}"
              .format(epoch + 1, epoch_num, reconstruction_loss.item(), kl_divergence.item()))

        acc_H2 = []
        nmi_H2 = []
        pur_H2 = []

        for i in range(10):
            kmeans = KMeans(n_clusters=max(np.int_(labels).flatten()))
            Y_pred_OK = kmeans.fit_predict(embedding)
            Y_pred_OK = np.array(Y_pred_OK)
            labels = np.array(labels)
            labels = labels.flatten()
            AM = clustering_metrics(Y_pred_OK, labels)
            acc, nmi, pur = AM.evaluationClusterModelFromLabel(print_msg=False)
            acc_H2.append(acc)
            nmi_H2.append(nmi)
            pur_H2.append(pur)
        print('ACC_LVGAE=', 100 * np.mean(acc_H2), '\n', 'NMI_LVGAE=', 100 * np.mean(nmi_H2), '\n', 'PUR_LVGAE=',
              100 * np.mean(pur_H2))

        acc_LVGAE_total.append(100 * np.mean(acc_H2))
        nmi_LVGAE_total.append(100 * np.mean(nmi_H2))
        pur_LVGAE_total.append(100 * np.mean(pur_H2))

        acc_LVGAE_total_std.append(100 * np.std(acc_H2))
        nmi_LVGAE_total_std.append(100 * np.std(nmi_H2))
        pur_LVGAE_total_std.append(100 * np.std(pur_H2))

        np.save(path_result + "{}.npy".format(epoch + 1), embedding)

    elif LINKPREDICTION and (epoch + 1) % 5 == 0:
        roc_score_temp, ap_score_temp = get_roc_score(test_edges, test_edges_false, embedding)
        roc_score.append(roc_score_temp)
        ap_score.append(ap_score_temp)

        print("Epoch: [{}]/[{}]".format(epoch + 1, epoch_num))
        print("AUC = {}".format(roc_score_temp))
        print("AP = {}".format(ap_score_temp))
##################################################  Result #############################################################
if CLUSTERING:
    index_max = np.argmax(acc_LVGAE_total)

    acc_LVGAE_max = np.float(acc_LVGAE_total[index_max])
    nmi_LVGAE_max = np.float(nmi_LVGAE_total[index_max])
    pur_LVGAE_max = np.float(pur_LVGAE_total[index_max])

    acc_std = np.float(acc_LVGAE_total_std[index_max])
    nmi_std = np.float(nmi_LVGAE_total_std[index_max])
    pur_std = np.float(pur_LVGAE_total_std[index_max])

    print('ACC_LVGAE_max={:.2f} +- {:.2f}'.format(acc_LVGAE_max, acc_std))
    print('NMI_LVGAE_max={:.2f} +- {:.2f}'.format(nmi_LVGAE_max, nmi_std))
    print('PUR_LVGAE_max={:.2f} +- {:.2f}'.format(pur_LVGAE_max, pur_std))


elif CLASSIFICATION:
    index_max = np.argmax(F1_score)
    print("LVGAE: F1-score_max is {:.2f}".format(100 * np.max(F1_score)))

elif LINKPREDICTION:
    print("VGAE: AUC_max is {:.2f}".format(100 * np.max(roc_score)))
    print("VGAE: AP_max is {:.2f}".format(100 * np.max(ap_score)))
########################################################### t- SNE #################################################
if TSNE:
    print("dataset is {}".format(DATASET))
    latent_representation_max = np.load(path_result + "{}.npy".format((index_max + 1) * 5))
    features = features.cpu().numpy()
    plot_embeddings(latent_representation_max, features, labels)
########################################################################################################################
end_time = time.time()
print("Running time is {}".format(end_time - start_time))
