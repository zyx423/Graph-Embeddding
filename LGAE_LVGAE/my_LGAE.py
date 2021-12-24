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

#  features: X (n × d); adjacency: similarity matrix; labels: Y
#  Parameters that need to be entered manually
######################################################### Setting #####################################################
DATASET = 'cora'
CLASSIFICATION = False
CLUSTERING = False
LINKPREDICTION = True
TSNE = False
path_result = "./embedding/LGAE/"
########################################## hyper-parameters##############################################################
epoch_num = 150
learning_rate = 1e-3
hidden_layer_1 = 128
################################### Load dataset   ######################################################################
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
acc_LGAE_total = []
nmi_LGAE_total = []
pur_LGAE_total = []

acc_LGAE_total_std = []
nmi_LGAE_total_std = []
pur_LGAE_total_std = []

F1_score = []

roc_score = []
ap_score = []
#################################################### Weight initialization  ###################################################
weight_mask = adjacency_matrix.view(-1) == 1
weight_tensor = torch.ones(weight_mask.size(0))
pos_weight = float(adjacency_matrix.shape[0] * adjacency_matrix.shape[0] - adjacency_matrix.sum()) / adjacency_matrix.sum()
weight_tensor[weight_mask] = 10

#######################################  Model #########################################################################
mse_loss = torch.nn.MSELoss(size_average=False)
bce_loss = torch.nn.BCELoss(size_average=False, weight = weight_tensor)
model_LGAE = myLGAE(features.shape[1], hidden_layer_1)
optimzer = torch.optim.Adam(model_LGAE.parameters(), lr=learning_rate)

start_time = time.time()
#######################################  Train and result ################################################################
for epoch in range(epoch_num):
    graph_reconstruction, embedding = model_LGAE(adjacency_convolution_kernel, features)
    loss = bce_loss(graph_reconstruction.view(-1), adjacency_matrix.view(-1))

    optimzer.zero_grad()
    loss.backward()
    optimzer.step()


    embedding = embedding.cpu().detach().numpy()
    if epoch == 150:
        torch.save(model_LGAE.state_dict(), 'mode_LGAE_save.pt')
    ##################################################### Results  ####################################################
    if CLASSIFICATION and (epoch + 1) % 5 == 0:
        print("Epoch:{},Loss:{:.4f}".format(epoch + 1, loss.item()))
        score = my_SVM(embedding, labels, 0.3)
        print("Epoch[{}/{}], score = {}".format(epoch + 1, epoch_num, score))
        F1_score.append(score)

    elif CLUSTERING and (epoch + 1) % 5 == 0:
        print("Epoch:{},Loss:{:.4f}".format(epoch+1, loss.item()))

        ACC_H2 = []
        NMI_H2 = []
        PUR_H2 = []
        ##############################################################################
        kmeans = KMeans(n_clusters=max(np.int_(labels).flatten()))
        for i in range(10):
            Y_pred_OK = kmeans.fit_predict(embedding)
            Labels_K = np.array(labels).flatten()
            AM = clustering_metrics(Y_pred_OK, Labels_K)
            ACC, NMI, PUR = AM.evaluationClusterModelFromLabel(print_msg=False)
            ACC_H2.append(ACC)
            NMI_H2.append(NMI)
            PUR_H2.append(PUR)
        print('ACC_H2=', 100 * np.mean(ACC_H2), '\n', 'NMI_H2=', 100 * np.mean(NMI_H2), '\n', 'PUR_H2=',
              100 * np.mean(PUR_H2))
        acc_LGAE_total.append(100 * np.mean(ACC_H2))
        nmi_LGAE_total.append(100 * np.mean(NMI_H2))
        pur_LGAE_total.append(100 * np.mean(PUR_H2))

        acc_LGAE_total_std.append(100 * np.std(ACC_H2))
        nmi_LGAE_total_std.append(100 * np.std(NMI_H2))
        pur_LGAE_total_std.append(100 * np.std(PUR_H2))

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
    index_max = np.argmax(acc_LGAE_total)

    ACC_LGAE_max = np.float(acc_LGAE_total[index_max])
    NMI_LGAE_max = np.float(nmi_LGAE_total[index_max])
    PUR_LGAE_max = np.float(pur_LGAE_total[index_max])

    ACC_STD = np.float(acc_LGAE_total_std[index_max])
    NMI_STD = np.float(nmi_LGAE_total_std[index_max])
    PUR_STD = np.float(pur_LGAE_total_std[index_max])

    print('ACC_LGAE_max={:.2f} +- {:.2f}'.format(ACC_LGAE_max, ACC_STD))
    print('NMI_LGAE_max={:.2f} +- {:.2f}'.format(NMI_LGAE_max, NMI_STD))
    print('PUR_LGAE_max={:.2f} +- {:.2f}'.format(PUR_LGAE_max, PUR_STD))

elif CLASSIFICATION:
    print("LVGAE: F1-score_max is {:.2f}".format(100 * np.max(F1_score)))

elif LINKPREDICTION:
    print("LGAE: AUC_max is {:.2f}".format(100 * np.max(roc_score)))
    print("LGAE: AP_max is {:.2f}".format(100 * np.max(ap_score)))
########################################################### t- SNE #################################################
if TSNE:
    print("dataset is {}".format(DATASET))
    latent_representation_max = np.load(path_result + "{}.npy".format((index_max + 1) * 5))
    features = features.cpu().numpy()
    plot_embeddings(latent_representation_max, features, labels)
########################################################################################################################

end_time = time.time()
print("Running time is {}".format(end_time - start_time))

# # print('END')
#summary.close()