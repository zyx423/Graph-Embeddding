from Data_Process import *
import torch

import sys
sys.path.append('D:\OneDrive - mail.nwpu.edu.cn\Optimal\Public\Python\Pre_Process')
from Data_Process import *
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

    def Encoder(self, Adjacency_Modified, H_0):
        H_1 = self.gconv1(torch.matmul(Adjacency_Modified, H_0))
        H_2 = self.gconv2(torch.matmul(Adjacency_Modified, H_1))
        return H_2

    def Graph_Decoder(self, H_2):
        graph_re = Graph_Construction(H_2)
        Graph_Reconstruction = graph_re.Middle()
        return Graph_Reconstruction


    def forward(self, Adjacency_Modified, H_0):
        Latent_Representation = self.Encoder(Adjacency_Modified, H_0)
        Graph_Reconstruction = self.Graph_Decoder(Latent_Representation)
        return Graph_Reconstruction, Latent_Representation

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

    def Encoder(self, Adjacency_Convolution, H_0):
        H_1 = self.gconv1((Adjacency_Convolution.mm(H_0)))
        H_2_mean = self.gconv2_mean(torch.matmul(Adjacency_Convolution, H_1))
        H_2_std = self.gconv2_std(torch.matmul(Adjacency_Convolution, H_1))
        return H_2_mean, H_2_std


    def Reparametrization(self, H_2_mean, H_2_std):
        eps = torch.randn_like(H_2_std)
        # H_2_std 并不是方差，而是：H_2_std = log(σ)
        std = torch.exp(H_2_std)
        # torch.randn 生成正态分布,这里是点乘
        Latent_Representation = eps.mul(std) + H_2_mean
        return Latent_Representation

    # 解码隐变量
    def Graph_Decoder(self, Latent_Representation):
        graph_re = Graph_Construction(Latent_Representation)
        Graph_Reconstruction = graph_re.Middle()
        return Graph_Reconstruction

    def forward(self, Adjacency_Convolution, H_0):
        H_2_mean, H_2_std = self.Encoder(Adjacency_Convolution, H_0)
        Latent_Representation = self.Reparametrization(H_2_mean, H_2_std)
        Graph_Reconstruction = self.Graph_Decoder(Latent_Representation)
        return Latent_Representation, Graph_Reconstruction, H_2_mean, H_2_std





