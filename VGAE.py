import sys
sys.path.append('./Data_Process')
path_result = "./Latent_representation/"
from Models import *
from Metrics import *
from Data_Process import *
from sklearn.cluster import KMeans
import numpy as np
import torch
import time

import warnings
warnings.filterwarnings('ignore')
######################################################### Setting #####################################################
Dataset = 'ATT'
Classification = True
Clustering = False
t_SNE = False
scale = 0
########################################## hyper-parameters##############################################################
Epoch_Num = 200
Learning_Rate = 5*1e-4

Hidden_Layer_1 = 1024
Hidden_Layer_2 = 128
################################### Load dataset   ######################################################################
if (Dataset is "cora") or (Dataset is "citeseer"):
    load_data = Load_Data(Dataset)
    Features, Labels, Adjacency_Matrix = load_data.Graph()
    Features = torch.Tensor(Features)
    Adjacency_Matrix = torch.Tensor(Adjacency_Matrix)
else:
    load_data = Load_Data(Dataset)
    Features, Labels = load_data.CPU()
    Features = torch.Tensor(Features)
################################### Calculate the adjacency matrix #########################################################
if('Adjacency_Matrix' in vars()):
    print('Adjacency matrix is raw')
    pass
else:
    print('Adjacency matrix is caculated by KNN')
    graph = Graph_Construction(Features)
    Adjacency_Matrix = graph.KNN()

################################################## convolution_kernel ##############################################
convolution_kernel = Convolution_Kernel(Adjacency_Matrix)
Adjacency_Convolution = convolution_kernel.Adjacency_Convolution()
############################################# is incomplate ############################################################
if scale != 0:
    ck = Convolution_Kernel(Adjacency_Matrix)
    Adjacency_Matrix = ck.Adjacency_Incomplete(scale)
    print("The incompleteness of the adjacency matrix is {}%".format(scale*100))
############################################ Results  Initialization ###################################################
ACC_VGAE_total = []
NMI_VGAE_total = []
PUR_VGAE_total = []

ACC_VGAE_total_STD = []
NMI_VGAE_total_STD = []
PUR_VGAE_total_STD = []

F1_score = []
#################################################### Weight initialization  ###################################################
weight_mask = Adjacency_Matrix.view(-1) == 1
weight_tensor = torch.ones(weight_mask.size(0))
pos_weight = float(Adjacency_Matrix.shape[0] * Adjacency_Matrix.shape[0] - Adjacency_Matrix.sum()) / Adjacency_Matrix.sum()
weight_tensor[weight_mask] = 80

#######################################################  Loss Function #################################################
def Loss_Function(Graph_Reconstruction, Graph_Raw, H_2_mean, H_2_std):
    bce_loss = torch.nn.BCELoss(size_average=False, weight = weight_tensor)
    Reconstruction_Loss = bce_loss(Graph_Reconstruction.view(-1), Graph_Raw.view(-1))
    KL_Divergence = -0.5 / Adjacency_Matrix.size(0) * (1 + 2 * H_2_std - H_2_mean ** 2 - torch.exp(H_2_std) ** 2).sum(1).mean()
    return Reconstruction_Loss, KL_Divergence

############################################## Model ###################################################################
model_VGAE = myVGAE(Features.shape[1], Hidden_Layer_1, Hidden_Layer_2)
optimzer = torch.optim.Adam(model_VGAE.parameters(), lr=Learning_Rate)
start_time = time.time()
#################################################### Train ###################################################
for epoch in range(Epoch_Num):
    Latent_Representation, Graph_Reconstruction, H_2_mean, H_2_std = model_VGAE(Adjacency_Convolution, Features)
    Reconstruction_Loss, KL_Divergence = Loss_Function(Graph_Reconstruction, Adjacency_Matrix, H_2_mean, H_2_std)
    loss = Reconstruction_Loss + KL_Divergence

    optimzer.zero_grad()
    loss.backward()
    optimzer.step()

    Latent_Representation = Latent_Representation.cpu().detach().numpy()
    ##################################################### Results  ####################################################
    if Classification and (epoch + 1) % 5 == 0:
        print("Epoch:{},Loss:{:.4f}".format(epoch + 1, loss.item()))
        score = mySVM(Latent_Representation, Labels, scale=0.3)
        print("Epoch[{}/{}], scale = {}".format(epoch + 1, Epoch_Num, score))
        F1_score.append(score)

    elif Clustering and (epoch + 1) % 5 == 0:
        print("Epoch[{}/{}], Reconstruction_Loss: {:.4f}, KL_Divergence: {:.4f}"
              .format(epoch + 1, Epoch_Num,  Reconstruction_Loss.item(), KL_Divergence.item()))

        ACC_H2 = []
        NMI_H2 = []
        PUR_H2 = []
        kmeans = KMeans(n_clusters=max(np.int_(Labels).flatten()))
        for i in range(10):
            Y_pred_OK = kmeans.fit_predict(Latent_Representation)
            Labels_K = np.array(Labels).flatten()
            AM = clustering_metrics(Y_pred_OK, Labels_K)
            ACC, NMI, PUR = AM.evaluationClusterModelFromLabel(print_msg=False)
            ACC_H2.append(ACC)
            NMI_H2.append(NMI)
            PUR_H2.append(PUR)

        print('ACC_VGAE=', 100 * np.mean(ACC_H2), '\n', 'NMI_VGAE=', 100 * np.mean(NMI_H2), '\n', 'PUR_VGAE=',
              100 * np.mean(PUR_H2))

        ACC_VGAE_total.append(100 * np.mean(ACC_H2))
        NMI_VGAE_total.append(100 * np.mean(NMI_H2))
        PUR_VGAE_total.append(100 * np.mean(PUR_H2))

        ACC_VGAE_total_STD.append(100 * np.std(ACC_H2))
        NMI_VGAE_total_STD.append(100 * np.std(NMI_H2))
        PUR_VGAE_total_STD.append(100 * np.std(PUR_H2))

        np.save(path_result + "{}.npy".format(epoch + 1), Latent_Representation)
##################################################  Result #############################################################
if Clustering:
    Index_MAX = np.argmax(ACC_VGAE_total)

    ACC_VGAE_max = np.float(ACC_VGAE_total[Index_MAX])
    NMI_VGAE_max = np.float(NMI_VGAE_total[Index_MAX])
    PUR_VGAE_max = np.float(PUR_VGAE_total[Index_MAX])

    ACC_STD = np.float(ACC_VGAE_total_STD[Index_MAX])
    NMI_STD = np.float(NMI_VGAE_total_STD[Index_MAX])
    PUR_STD = np.float(PUR_VGAE_total_STD[Index_MAX])

    print('ACC_VGAE_max={:.2f} +- {:.2f}'.format(ACC_VGAE_max, ACC_STD))
    print('NMI_VGAE_max={:.2f} +- {:.2f}'.format(NMI_VGAE_max, NMI_STD))
    print('PUR_VGAE_max={:.2f} +- {:.2f}'.format(PUR_VGAE_max, PUR_STD))
    print("The incompleteness of the adjacency matrix is {}%".format(scale * 100))

elif Classification:
    print("VGAE: F1-score_max is {:.2f}".format(100 * np.max(F1_score)))
########################################################### t- SNE #################################################
if t_SNE:
    print("dataset is {}".format(dataset))
    Latent_Representation_max = np.load(path_result + "{}.npy".format((Index_MAX+1) * 5))
    Features = np.array(Features)
    plot_embeddings(Latent_Representation_max, Features, Labels)
########################################################################################################################
end_time = time.time()
print("Running time is {}".format(end_time - start_time))
