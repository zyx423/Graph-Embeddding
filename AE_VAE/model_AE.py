import sys
sys.path.append('../functions')
from data_process import *
import torch

class MyAE(torch.nn.Module):
    def __init__(self, d_0, d_1, d_2, d_3, d_4):
        super(MyAE, self).__init__()

        self.conv1 = torch.nn.Sequential(
            torch.nn.Linear(d_0, d_1),
            torch.nn.ReLU(inplace=True)
        )

        self.conv2 = torch.nn.Sequential(
            torch.nn.Linear(d_1, d_2),
        )

        self.conv3 = torch.nn.Sequential(
            torch.nn.Linear(d_2, d_3),
            torch.nn.ReLU(inplace=True)
        )

        self.conv4 = torch.nn.Sequential(
            torch.nn.Linear(d_3, d_4),
            torch.nn.Tanh()
        )
    def encoder(self, H_0):
        H_1 = self.conv1(H_0)
        H_2 = self.conv2(H_1)
        return H_2

    def decoder(self, H_2):
        H_3 = self.conv3(H_2)
        H_4 = self.conv4(H_3)
        return H_4

    def forward(self, H_0):
        Latent_Representation = self.encoder(H_0)
        Features_Reconstrction = self.decoder(Latent_Representation)
        return Latent_Representation, Features_Reconstrction

class MyVAE(torch.nn.Module):
    def __init__(self, d_0, d_1, d_2, d_3, d_4):
        super(MyVAE, self).__init__()

        self.conv1 = torch.nn.Sequential(

            torch.nn.Linear(d_0, d_1),
            torch.nn.ReLU(inplace=True)
        )

        # VAE有两个encoder，一个用来学均值，一个用来学方差
        self.conv2_mean = torch.nn.Sequential(
            torch.nn.Linear(d_1, d_2)

        )
        self.conv2_std = torch.nn.Sequential(
            torch.nn.Linear(d_1, d_2)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Linear(d_2, d_3),
            torch.nn.ReLU()
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Linear(d_3, d_4),
            torch.nn.Tanh()
        )

    def encoder(self, H_0):
        H_1 = self.conv1(H_0)
        H_2_mean = self.conv2_mean(H_1)
        H_2_std = self.conv2_std(H_1)
        return H_2_mean, H_2_std

    def reparametrization(self, H_2_mean, H_2_std):
        # randn 就是标准正态分布， rand就是{0,1}之间的均匀分布
        eps = torch.rand_like(H_2_std)
        # H_2_std 并不是方差，而是：H_2_std = log(σ)
        std = torch.exp(H_2_std)
        latent_representation = eps * std + H_2_mean
        return latent_representation

    # 解码隐变量
    def decoder(self, Latent_Representation):
        H_3 = self.conv3(Latent_Representation)
        Features_Reconstruction = self.conv4(H_3)
        return Features_Reconstruction

    # 计算重构值和隐变量z的分布参数
    def forward(self, H_0):
        H_2_mean, H_2_std = self.encoder(H_0)
        Latent_Representation = self.reparametrization(H_2_mean, H_2_std)
        Features_Reconstruction = self.decoder(Latent_Representation)
        return Latent_Representation, Features_Reconstruction, H_2_mean, H_2_std



class MySAE(torch.nn.Module):
    def __init__(self, input_dim, middle_dim, output_dim, bias=False):
        super(MySAE, self).__init__()

        # 直接把encoder和decoder写在这里也可以，网络结构比较简单
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, middle_dim),
            torch.nn.ReLU(inplace=True)
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(middle_dim, output_dim),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, H_0):
        latent_representation = self.encoder(H_0)
        features_reconstrction = self.decoder(latent_representation)
        return latent_representation, features_reconstrction