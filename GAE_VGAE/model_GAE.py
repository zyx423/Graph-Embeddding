import sys
sys.path.append('../functions')
from data_process import *
import torch


class myGAE(torch.nn.Module):
    def __init__(self, d_0, d_1, d_2):
        super(myGAE, self).__init__()

        self.gconv1 = torch.nn.Sequential(
            torch.nn.Linear(d_0, d_1),
            torch.nn.ReLU(inplace=True)
        )
        self.gconv1[0].weight.data = get_weight_initial(d_1, d_0)

        self.gconv2 = torch.nn.Sequential(
            torch.nn.Linear(d_1, d_2),
            # torch.nn.Dropout(0.5)
        )
        self.gconv2[0].weight.data = get_weight_initial(d_2, d_1)

    def encoder(self, Adjacency_Modified, H_0):
        H_1 = self.gconv1(torch.matmul(Adjacency_Modified, H_0))
        H_2 = self.gconv2(torch.matmul(Adjacency_Modified, H_1))
        return H_2

    def graph_decoder(self, H_2):
        graph_re = GraphConstruction(H_2)
        graph_reconstruction = graph_re.middle()
        return graph_reconstruction


    def forward(self, adjacency_modified, H_0):
        latent_representation = self.encoder(adjacency_modified, H_0)
        graph_reconstruction = self.graph_decoder(latent_representation)
        return graph_reconstruction, latent_representation

class myVGAE(torch.nn.Module):
    def __init__(self, d_0, d_1, d_2):
        super(myVGAE, self).__init__()

        self.gconv1 = torch.nn.Sequential(
            torch.nn.Linear(d_0, d_1),
            torch.nn.ReLU(inplace=True)
        )
        self.gconv1[0].weight.data = get_weight_initial(d_1, d_0)

        self.gconv2_mean = torch.nn.Sequential(
            torch.nn.Linear(d_1, d_2)
        )
        self.gconv2_mean[0].weight.data = get_weight_initial(d_2, d_1)

        self.gconv2_std = torch.nn.Sequential(
            torch.nn.Linear(d_1, d_2)
        )
        self.gconv2_std[0].weight.data = get_weight_initial(d_2, d_1)

    def encoder(self, Adjacency_Convolution, H_0):
        H_1 = self.gconv1((Adjacency_Convolution.mm(H_0)))
        H_2_mean = self.gconv2_mean(torch.matmul(Adjacency_Convolution, H_1))
        H_2_std = self.gconv2_std(torch.matmul(Adjacency_Convolution, H_1))
        return H_2_mean, H_2_std


    def reparametrization(self, H_2_mean, H_2_std):
        eps = torch.randn_like(H_2_std)
        # H_2_std 并不是方差，而是：H_2_std = log(σ)
        std = torch.exp(H_2_std)
        # torch.randn 生成正态分布,这里是点乘
        latent_representation = eps.mul(std) + H_2_mean
        return latent_representation

    # 解码隐变量
    def graph_decoder(self, latent_representation):
        graph_re = GraphConstruction(latent_representation)
        graph_reconstruction = graph_re.middle()
        return graph_reconstruction

    def forward(self, adjacency_convolution, H_0):
        H_2_mean, H_2_std = self.encoder(adjacency_convolution, H_0)
        latent_representation = self.reparametrization(H_2_mean, H_2_std)
        graph_reconstruction = self.graph_decoder(latent_representation)
        return latent_representation, graph_reconstruction, H_2_mean, H_2_std