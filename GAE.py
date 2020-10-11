from sklearn.datasets import make_moons
import sys
sys.path.append('./Data_Process')
path_result = "./Latent_representation/"
from Models import *
from Metrics import *
import scipy.io as scio
from Data_Process import *
from sklearn.cluster import KMeans
import numpy as np
import torch
import time

import warnings
warnings.filterwarnings('ignore')

#  Features: X (n × d); adjacency: similarity matrix; labels: Y
#  Parameters that need to be entered manually
######################################################### Setting #####################################################
Dataset = 'cora'
Classification = True
Clustering = False
t_SNE = False
scale = 0
########################################## hyper-parameters##############################################################
Epoch_Num = 200
Learning_Rate = 1e-4

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

################################################ adjacency convolution ##################################################

convolution_kernel = Convolution_Kernel(Adjacency_Matrix)
Adjacency_Convolution = convolution_kernel.Adjacency_Convolution()
############################################ Results  Initialization ###################################################
ACC_GAE_total = []
NMI_GAE_total = []
PUR_GAE_total = []

ACC_GAE_total_STD = []
NMI_GAE_total_STD = []
PUR_GAE_total_STD = []

F1_score = []
#######################################  Model #########################################################################
mse_loss = torch.nn.MSELoss(size_average=False)
bce_loss = torch.nn.BCELoss(size_average=False)
model_GAE = myGAE(Features.shape[1], Hidden_Layer_1, Hidden_Layer_2)
optimzer = torch.optim.Adam(model_GAE.parameters(), lr=Learning_Rate)

start_time = time.time()
#######################################  Train and result ################################################################
for epoch in range(Epoch_Num):
    Graph_Reconstruction, Latent_Representation = model_GAE(Adjacency_Convolution, Features)
    loss = bce_loss(Graph_Reconstruction.view(-1), Adjacency_Matrix.view(-1))


    optimzer.zero_grad()
    loss.backward()
    optimzer.step()

    Latent_Representation = Latent_Representation.cpu().detach().numpy()

    ##################################################### Results  ####################################################
    if Classification and (epoch + 1) % 5 == 0:
        print("Epoch:{},Loss:{:.4f}".format(epoch + 1, loss.item()))
        score = mySVM(Latent_Representation, Labels, scale=0.3)
        print("Epoch[{}/{}], F1-score = {}".format(epoch + 1, Epoch_Num, score))
        np.save(path_result + "{}.npy".format(epoch + 1), Latent_Representation)
        F1_score.append(score)

    elif Clustering and (epoch + 1) % 5 == 0:
        print("Epoch:{},Loss:{:.4f}".format(epoch + 1, loss.item()))
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

        ACC_GAE_total.append(100 * np.mean(ACC_H2))
        NMI_GAE_total.append(100 * np.mean(NMI_H2))
        PUR_GAE_total.append(100 * np.mean(PUR_H2))
        ACC_GAE_total_STD.append(100 * np.std(ACC_H2))
        NMI_GAE_total_STD.append(100 * np.std(NMI_H2))
        PUR_GAE_total_STD.append(100 * np.std(PUR_H2))

        print('ACC_H2=', 100 * np.mean(ACC_H2), '\n', 'NMI_H2=', 100 * np.mean(NMI_H2), '\n', 'PUR_H2=',
              100 * np.mean(PUR_H2))

        np.save(path_result + "{}.npy".format(epoch + 1), Latent_Representation)
    ###############################################################Clustering  Result ##############################
if Clustering:
    Index_MAX = np.argmax(ACC_GAE_total)
    ACC_GAE_max = np.float(ACC_GAE_total[Index_MAX])
    NMI_GAE_max = np.float(NMI_GAE_total[Index_MAX])
    PUR_GAE_max = np.float(PUR_GAE_total[Index_MAX])

    ACC_STD = np.float(ACC_GAE_total_STD[Index_MAX])
    NMI_STD = np.float(NMI_GAE_total_STD[Index_MAX])
    PUR_STD = np.float(PUR_GAE_total_STD[Index_MAX])

    print('ACC_GAE_max={:.2f} +- {:.2f}'.format(ACC_GAE_max, ACC_STD))
    print('NMI_GAE_max={:.2f} +- {:.2f}'.format(NMI_GAE_max, NMI_STD))
    print('PUR_GAE_max={:.2f} +- {:.2f}'.format(PUR_GAE_max, PUR_STD))
    print("The incompleteness of the adjacency matrix is {}%".format(scale * 100))

elif Classification:
    Index_MAX = np.argmax(F1_score)
    print("GAE: F1-score_max is {:.2f}".format(100*np.max(F1_score)))
########################################################### t- SNE #################################################
if t_SNE:
    print("dataset is {}".format(Dataset))
    print("Index_Max = {}".format(Index_MAX))
    Latent_Representation_max = np.load(path_result + "{}.npy".format((Index_MAX+1) * 5))
    Features = np.array(Features)
    plot_embeddings(Latent_Representation_max, Features, Labels)

########################################################################################################################
end_time = time.time()
print("Running time is {}".format(end_time - start_time))

