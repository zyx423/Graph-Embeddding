import sys
sys.path.append('../functions')
from data_process import *
import torch

class myLGAE(torch.nn.Module):
    def __init__(self, d_0, d_1):
        super(myLGAE, self).__init__()

        self.gconv1 = torch.nn.Sequential(
            torch.nn.Linear(d_0, d_1),
        )
        self.gconv1[0].weight.data = get_weight_initial(d_1, d_0)

    def encoder(self, Adjacency_Modified, H_0):
        H_1 = self.gconv1(torch.matmul(Adjacency_Modified, H_0))
        return H_1

    def graph_decoder(self, H_1):
        graph_re = GraphConstruction(H_1)
        graph_reconstruction = graph_re.middle()
        return graph_reconstruction


    def forward(self, Adjacency_Modified, H_0):
        latent_representation = self.encoder(Adjacency_Modified, H_0)
        graph_reconstruction = self.graph_decoder(latent_representation)
        return graph_reconstruction, latent_representation


class myLVGAE(torch.nn.Module):
    def __init__(self, d_0, d_1):
        super(myLVGAE, self).__init__()

        self.gconv1_mean = torch.nn.Sequential(
            torch.nn.Linear(d_0, d_1)
        )
        self.gconv1_mean[0].weight.data = get_weight_initial(d_1, d_0)

        self.gconv1_std = torch.nn.Sequential(
            torch.nn.Linear(d_0, d_1)
        )
        self.gconv1_std[0].weight.data = get_weight_initial(d_1, d_0)

    def encoder(self, adjacency_convolution, H_0):
        H_1_mean = self.gconv1_mean(torch.matmul(adjacency_convolution, H_0))
        H_1_std = self.gconv1_std(torch.matmul(adjacency_convolution, H_0))
        return H_1_mean, H_1_std

    def reparametrization(self, H_1_mean, H_1_std):
        eps = torch.randn_like(H_1_std)
        # H_1_std 并不是方差，而是：H_1_std = log(σ)
        std = torch.exp(H_1_std)
        # torch.randn 生成正态分布,这里是点乘
        latent_representation = eps.mul(std) + H_1_mean
        return latent_representation

    # 解码隐变量
    def graph_decoder(self, latent_representation):
        graph_re = GraphConstruction(latent_representation)
        graph_reconstruction = graph_re.middle()
        return graph_reconstruction

    def forward(self, adjacency_convolution, H_0):
        H_1_mean, H_1_std = self.encoder(adjacency_convolution, H_0)
        latent_representation = self.reparametrization(H_1_mean, H_1_std)
        graph_reconstruction = self.graph_decoder(latent_representation)
        return latent_representation, graph_reconstruction, H_1_mean, H_1_std