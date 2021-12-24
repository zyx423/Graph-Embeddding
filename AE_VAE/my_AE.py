
from model_AE import *
from metrics import *
from data_process import *
from sklearn.cluster import KMeans
import numpy as np
import torch
import time
#from tensorboardX import SummaryWriter
import warnings
warnings.filterwarnings('ignore')
path_result = "./embedding/AE/"

######################################################### Setting #################################################
DATASET = 'ORL'
CLASSIFICATION = True
CLUSTERING = False
TSNE = False

######################################  Load dataset  ###############################################################
load_data = LoadData(DATASET)
features, labels = load_data.mat()

########################################  Normalization  ##########################################################
norm = Normalized(features)
features = norm.MinMax()
features = torch.Tensor(features)
########################################### hyper-parameters##########################################
epoch_num = 200
learning_rate = 1e-3

input_dim = features.shape[1]
hidden_layer_1 = 512
hidden_layer_2 = 128
hidden_layer_3 = hidden_layer_1
output_dim = input_dim

############################################ Results  Initialization ######################################
acc_AE_total = []
nmi_AE_total = []
pur_AE_total = []

acc_AE_total_std = []
nmi_AE_total_std = []
pur_AE_total_std = []

F1_score = []
########################################################  Loss function ##############################################
loss_fn = torch.nn.MSELoss(size_average=False)
model_AE = MyAE(input_dim, hidden_layer_1, hidden_layer_2, hidden_layer_3, output_dim)
optimzer = torch.optim.Adam(model_AE.parameters(), lr=learning_rate)

############################################################ Training ###################################################
start_time = time.time()
for epoch in range(epoch_num):

    embedding, features_reconstrction = model_AE(features)
    loss = loss_fn(features, features_reconstrction)

    optimzer.zero_grad()
    loss.backward()
    optimzer.step()

    embedding = embedding.cpu().detach().numpy()
    ##################################################### Results  ####################################################
    if CLASSIFICATION and (epoch + 1) % 5 == 0:
        print("Epoch:{},Loss:{:.4f}".format(epoch + 1, loss.item()))
        score = my_SVM(embedding, labels, 0.3)
        print("Epoch[{}/{}], scale = {}".format(epoch + 1, epoch_num, score))
        np.save(path_result + "{}.npy".format(epoch + 1), embedding)
        F1_score.append(score)

    elif CLUSTERING and (epoch + 1) % 5 == 0:

            print("Epoch:{},Loss:{:.4f}".format(epoch+1, loss.item()))
            embedding = embedding.cpu().detach().numpy()

            ACC_H2 = []
            NMI_H2 = []
            PUR_H2 = []
            for i in range(10):
                kmeans = KMeans(n_clusters=max(np.int_(labels).flatten()))
                Y_pred_OK = kmeans.fit_predict(embedding)
                Y_pred_OK = np.array(Y_pred_OK)
                labels = np.array(labels).flatten()
                AM = clustering_metrics(Y_pred_OK, labels)
                ACC, NMI, PUR = AM.evaluationClusterModelFromLabel(print_msg=False)
                ACC_H2.append(ACC)
                NMI_H2.append(NMI)
                PUR_H2.append(PUR)

            print(f'ACC_H2=', 100 * np.mean(ACC_H2), '\n', 'NMI_H2=', 100 * np.mean(NMI_H2), '\n', 'PUR_H2=',
                  100 * np.mean(PUR_H2))

            acc_AE_total.append(100 * np.mean(ACC_H2))
            nmi_AE_total.append(100 * np.mean(NMI_H2))
            pur_AE_total.append(100 * np.mean(PUR_H2))

            acc_AE_total_std.append(100 * np.std(ACC_H2))
            nmi_AE_total_std.append(100 * np.std(NMI_H2))
            pur_AE_total_std.append(100 * np.std(PUR_H2))
            np.save(path_result + "{}.npy".format(epoch + 1), embedding)
##################################################  Result #############################################################
if CLUSTERING:

    index_max = np.argmax(acc_AE_total)

    ACC_AE_max = np.float(acc_AE_total[index_max])
    NMI_AE_max = np.float(nmi_AE_total[index_max])
    PUR_AE_max = np.float(pur_AE_total[index_max])

    ACC_STD = np.float(acc_AE_total_std[index_max])
    NMI_STD = np.float(nmi_AE_total_std[index_max])
    PUR_STD = np.float(pur_AE_total_std[index_max])

    print('ACC_AE_max={:.2f} +- {:.2f}'.format(ACC_AE_max, ACC_STD))
    print('NMI_AE_max={:.2f} +- {:.2f}'.format(NMI_AE_max, NMI_STD))
    print('PUR_AE_max={:.2f} +- {:.2f}'.format(PUR_AE_max, PUR_STD))

elif CLASSIFICATION:
    index_max = np.argmax(F1_score)
    print("AE: F1-score_max is {:.2f}".format(100*np.max(F1_score)))

########################################################### t- SNE #################################################
if TSNE:
    print("dataset is {}".format(DATASET))
    print("Index_Max = {}".format(index_max))
    latent_representation_max = np.load(path_result + "{}.npy".format((index_max + 1) * 5))
    features = np.array(features)
    plot_embeddings(latent_representation_max, features, labels)

########################################################################################################################
end_time = time.time()
print("Running time is {}".format(end_time - start_time))
