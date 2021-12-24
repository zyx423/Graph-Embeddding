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
path_result = "./embedding/VAE/"
######################################################### Setting #################################################
DATASET = 'ATT'
CLASSIFICATION = True
CLUSTERING = False
TSNE = False

#######################################Load dataset  ##############################################################
load_data = LoadData(DATASET)
features, labels = load_data.mat()

###  Normalization
norm = Normalized(features)
features = norm.Normal()
features = torch.Tensor(features)
################################################## parameters ######################################################
epoch_num = 200
learning_rate = 1e-3

hidden_layer_1 = 512
hidden_layer_2 = 128
hidden_layer_3 = 512

batch_n = features.shape[0]
input_dim = features.shape[1]
output_dim = input_dim

################################################# Result Initialization ################################################
acc_VAE_total = []
nmi_VAE_total = []
pur_VAE_total = []

acc_VAE_total_std = []
nmi_VAE_total_std = []
pur_VAE_total_std = []

F1_score = []
################################################  Loss_Function ########################################################
def Loss_Function(features_reconstruction, features, H_2_mean, H_2_std):

    re_loss = torch.nn.MSELoss(size_average=False)
    reconstruction_loss = re_loss(features_reconstruction, features)
    KLD_element = 1 + 2 * H_2_std - H_2_mean.pow(2) - H_2_std.exp() ** 2
    kl_divergence = torch.sum(KLD_element).mul_(-0.5)
    return reconstruction_loss, kl_divergence

###############################################  Model ###############################################################
model_VAE = MyVAE(input_dim, hidden_layer_1, hidden_layer_2, hidden_layer_3, output_dim)
optimzer = torch.optim.Adam(model_VAE.parameters(), lr=learning_rate)

for epoch in range(epoch_num):
    embedding, features_reconstruction, H_2_mean, H_2_std = model_VAE(features)
    reconstruction_loss, KL_divergence = Loss_Function(features_reconstruction, features, H_2_mean, H_2_std)
    loss = reconstruction_loss + KL_divergence

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
        print("Epoch[{}/{}], Reconstruction_Loss: {:.4f}, KL_Divergence: {:.4f}"
              .format(epoch + 1, epoch_num, reconstruction_loss.item(), KL_divergence))


        acc_H2 = []
        nmi_H2 = []
        pur_H2 = []
        for i in range(5):
            kmeans = KMeans(n_clusters=max(np.int_(labels).flatten()))
            Y_pred_OK = kmeans.fit_predict(embedding)
            Y_pred_OK = np.array(Y_pred_OK)
            labels = np.array(labels)
            labels = labels.flatten()
            AM = clustering_metrics(Y_pred_OK, labels)
            ACC, NMI, PUR = AM.evaluationClusterModelFromLabel(print_msg=False)
            acc_H2.append(ACC)
            nmi_H2.append(NMI)
            pur_H2.append(PUR)
        print('ACC_H2=', 100 * np.mean(acc_H2), '\n', 'NMI_H2=', 100 * np.mean(nmi_H2), '\n', 'PUR_H2=',
              100 * np.mean(pur_H2))
        acc_VAE_total.append(100 * np.mean(acc_H2))
        nmi_VAE_total.append(100 * np.mean(nmi_H2))
        pur_VAE_total.append(100 * np.mean(pur_H2))

        acc_VAE_total_std.append(100 * np.std(acc_H2))
        nmi_VAE_total_std.append(100 * np.std(nmi_H2))
        pur_VAE_total_std.append(100 * np.std(pur_H2))


##################################################  Result ##################################################
if CLUSTERING:
    index_max = np.argmax(acc_VAE_total)

    acc_VAE_max = np.float(acc_VAE_total[index_max])
    nmi_VAE_max = np.float(nmi_VAE_total[index_max])
    pur_VAE_max = np.float(pur_VAE_total[index_max])

    acc_std = np.float(acc_VAE_total_std[index_max])
    nmi_std = np.float(nmi_VAE_total_std[index_max])
    pur_std = np.float(pur_VAE_total_std[index_max])

    print('ACC_VAE_max={:.2f} +- {:.2f}'.format(acc_VAE_max, acc_std))
    print('NMI_VAE_max={:.2f} +- {:.2f}'.format(nmi_VAE_max, nmi_std))
    print('PUR_VAE_max={:.2f} +- {:.2f}'.format(pur_VAE_max, pur_std))

elif CLASSIFICATION:
    index_max = np.argmax(F1_score)
    print("VAE: F1-score_max is {:.2f}".format(100 * np.max(F1_score)))
    ########################################################## t- SNE #################################################

if TSNE:
    print("dataset is {}".format(DATASET))
    print("Index_Max = {}".format(index_max))
    latent_representation_max = np.load(path_result + "{}.npy".format((index_max + 1) * 5))
    features = np.array(features)
    plot_embeddings(latent_representation_max, features, labels)