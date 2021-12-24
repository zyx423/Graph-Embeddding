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
############################################### Dataset #####################################################################
#  features: X (n × d); adjacency: similarty matrix; labels: Y
#  Parameters that need to be entered manually
DATASET = 'ATT'
load_data = LoadData(DATASET)
features, labels = load_data.mat()

############################################# Parameters ##############################################################
normalized = Normalized(features)
features = normalized.Normal()
features = torch.Tensor(features)

############################################ hyper-parameters ##########################################################
lambda_1 = 0.01
epoch_num = 80
learning_rate = 1e-5
hidden_layer_1 = 500
hidden_layer_2 = 256

sample_num = features.shape[0]
input_dim = features.shape[1]
output_dim = input_dim

#############################################  First Stack  ##############################################################
loss_fn = torch.nn.MSELoss()
model_SAE_1 = MySAE(input_dim, hidden_layer_1, output_dim)
optimzer_1 = torch.optim.Adam(model_SAE_1.parameters(), lr=learning_rate)

# if torch.cuda.is_available():
#    model_SAE_1.cuda()
#    features = features.cuda()

acc_SAE_total = []
nmi_SAE_total = []
pur_SAE_total = []

for epoch in range(epoch_num):
    embedding_1, features_reconstruction_1 = model_SAE_1(features)
    loss_1 = loss_fn(features, features_reconstruction_1)

    optimzer_1.zero_grad()
    loss_1.backward(retain_graph=False)
    optimzer_1.step()
    #summary.add_scalar('loss: ', loss.item(), epoch)

################################################# Second Stack  #########################################################
features_2 = embedding_1.type(features.type())
# Importance: must detach the gradient of Features_2
features_2 = features_2.detach()

model_SAE_2 = MySAE(hidden_layer_1, hidden_layer_2, hidden_layer_1)
optimzer_2 = torch.optim.Adam(model_SAE_2.parameters(), lr=learning_rate)

# if torch.cuda.is_available():
#    model_SAE_2.cuda()
#    features_2 = features_2.cuda()


for epoch in range(epoch_num):
    embedding_2, features_reconstruction_2 = model_SAE_2(features_2)
    loss_2 = loss_fn(features_2, features_reconstruction_2)

    optimzer_2.zero_grad()
    loss_2.backward(retain_graph=False)
    optimzer_2.step()

    if epoch < 4 \
            or (epoch < epoch_num and (epoch + 1) % 10 == 0):

        print("Epoch:{},Loss:{:.4f}".format(epoch + 1, loss_2.item()))
        embedding_2 = embedding_2.cpu().detach().numpy()

        ACC_H2 = []
        NMI_H2 = []
        PUR_H2 = []
        for i in range(10):
            kmeans = KMeans(n_clusters=max(np.int_(labels).flatten()))
            Y_pred_OK = kmeans.fit_predict(embedding_2)
            Y_pred_OK = np.array(Y_pred_OK)
            labels = np.array(labels)
            labels = labels.flatten()
            AM = clustering_metrics(Y_pred_OK, labels)
            ACC, NMI, PUR = AM.evaluationClusterModelFromLabel(print_msg=False)
            ACC_H2.append(ACC)
            NMI_H2.append(NMI)
            PUR_H2.append(PUR)
        print(f'ACC_H2=', 100 * np.mean(ACC_H2), '\n', 'NMI_H2=', 100 * np.mean(NMI_H2), '\n', 'PUR_H2=',
              100 * np.mean(PUR_H2))
        acc_SAE_total.append(100 * np.mean(ACC_H2))
        nmi_SAE_total.append(100 * np.mean(NMI_H2))
        pur_SAE_total.append(100 * np.mean(PUR_H2))

acc_SAE_max = np.max(acc_SAE_total)
nmi_SAE_max = np.max(nmi_SAE_total)
pur_SAE_max = np.max(pur_SAE_total)
print('ACC_SAE_max={:.2f}'.format(acc_SAE_max))
print('NMI_SAE_max={:.2f}'.format(nmi_SAE_max))
print('PUR_SAE_max={:.2f}'.format(pur_SAE_max))
