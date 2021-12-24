import sys
sys.path.append('../functions')
from data_process import *
import torch

class mySDNE(torch.nn.Module):
    def __init__(self, d_0, d_1, d_2):
        super(mySDNE, self).__init__()

        self.conv1 = torch.nn.Sequential(
            torch.nn.Linear(d_0, d_1),
            torch.nn.ReLU(inplace=True)
        )

        self.conv2 = torch.nn.Sequential(
            torch.nn.Linear(d_1, d_2)
        )

        self.conv3 = torch.nn.Sequential(
            torch.nn.Linear(d_2, d_1),
            torch.nn.ReLU(inplace=True)
        )

        self.conv4 = torch.nn.Sequential(
            torch.nn.Linear(d_1, d_0),
            torch.nn.Sigmoid()
        )

    def Encoder(self, H_0):
        H_1 = self.conv1(H_0)
        H_2 = self.conv2(H_1)
        return H_2

    def Decoder(self, H_2):
        H_3 = self.conv3(H_2)
        H_4 = self.conv4(H_3)
        return H_4

    def forward(self, G_0):
        Latent_Representation = self.Encoder(G_0)
        Graph_Reconstrction = self.Decoder(Latent_Representation)
        return Latent_Representation, Graph_Reconstrction
