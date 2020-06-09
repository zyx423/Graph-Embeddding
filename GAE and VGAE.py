import torch

class myAE(torch.nn.Module):
    def __init__(self, d_0, d_1, d_2, d_3, d_4):
        super(myAE, self).__init__()

        self.conv1 = torch.nn.Sequential(
            torch.nn.Linear(d_0, d_1, bias=False),
            torch.nn.ReLU(inplace=True)
        )

        self.conv2 = torch.nn.Sequential(
            torch.nn.Linear(d_1, d_2, bias=False),
            torch.nn.ReLU(inplace=True)
        )

        self.conv3 = torch.nn.Sequential(
            torch.nn.Linear(d_2, d_3, bias=False),
            torch.nn.ReLU(inplace=True)
        )

        self.conv4 = torch.nn.Sequential(
            torch.nn.Linear(d_3, d_4, bias=False),
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

    def forward(self, H_0):
        Latent_Representation = self.Encoder(H_0)
        Features_Reconstrction = self.Decoder(Latent_Representation)
        return Latent_Representation, Features_Reconstrction

class myVAE(torch.nn.Module):
    def __init__(self, d_0, d_1, d_2, d_3, d_4, bias=False):
        super(myVAE, self).__init__()

        self.conv1 = torch.nn.Sequential\
        (
            torch.nn.Linear(d_0, d_1, bias= False),
            torch.nn.ReLU(inplace=True)
        )

        # VAE有两个encoder，一个用来学均值，一个用来学方差
        self.conv2_mean = torch.nn.Sequential(
            torch.nn.Linear(d_1, d_2, bias=False)

        )
        self.conv2_std = torch.nn.Sequential(
            torch.nn.Linear(d_1, d_2, bias=False)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Linear(d_2, d_3, bias=False),
            torch.nn.ReLU(inplace=False)
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Linear(d_3, d_4, bias=False),
            torch.nn.Sigmoid()
        )

    def Encoder(self, H_0):
        H_1 = self.conv1(H_0)
        H_2_mean = self.conv2_mean(H_1)
        H_2_std = self.conv2_std(H_1)
        return H_2_mean, H_2_std

    def Reparametrization(self, H_2_mean, H_2_std):
        # sigma = 0.5*exp(log(sigma^2))= 0.5*exp(log(var))
        std = 0.5 * torch.exp(H_2_std)
        # N(mu, std^2) = N(0, 1) * std + mu。
        # 数理统计中的正态分布方差，刚学过， std是方差。
        # torch.randn 生成正态分布
        Latent_Representation = torch.randn(std.size()) * std + H_2_mean
        return Latent_Representation

    # 解码隐变量
    def Decoder(self, Latent_Representation):
        H_3 = self.conv3(Latent_Representation)
        Features_Reconstruction = self.conv4(H_3)
        return Features_Reconstruction

    # 计算重构值和隐变量z的分布参数
    def forward(self, H_0):
        H_2_mean, H_2_std = self.Encoder(H_0)
        Latent_Representation = self.Reparametrization(H_2_mean, H_2_std)
        Features_Reconstruction = self.Decoder(Latent_Representation)
        return Latent_Representation, Features_Reconstruction, H_2_mean, H_2_std


class myGAE(torch.nn.Module):
    def __init__(self, d_0, d_1, d_2):
        super(myGAE, self).__init__()

        self.gconv1 = torch.nn.Sequential(
            torch.nn.Linear(d_0, d_1, bias=False),
            torch.nn.ReLU(inplace=True)
        )


        self.gconv2 = torch.nn.Sequential(
            torch.nn.Linear(d_1, d_2, bias=False)
        )


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
            torch.nn.Linear(d_0, d_1, bias=False),
            torch.nn.ReLU(inplace=True)
        )
        # self.gconv1[0].weight.data = get_weight_initial(d_1, d_0)

        self.gconv2_mean = torch.nn.Sequential(
            torch.nn.Linear(d_1, d_2, bias=False)
        )
        # self.gconv2_mean[0].weight.data = get_weight_initial(d_2, d_1)

        self.gconv2_std = torch.nn.Sequential(
            torch.nn.Linear(d_1, d_2, bias=False)
        )
        # self.gconv2_std[0].weight.data = get_weight_initial(d_2, d_1)

    def Encoder(self, Adjacency_Modified, H_0):
        H_1 = self.gconv1(torch.matmul(Adjacency_Modified, H_0))
        H_2_mean = self.gconv2_mean(torch.matmul(Adjacency_Modified, H_1))
        H_2_std = self.gconv2_std(torch.matmul(Adjacency_Modified, H_1))
        return H_2_mean, H_2_std

    def Reparametrization(self, H_2_mean, H_2_std):

        # sigma = 0.5*exp(log(sigma^2))= 0.5*exp(log(var))
        std = 0.5 * torch.exp(H_2_std)
        # N(mu, std^2) = N(0, 1) * std + mu。
        # 数理统计中的正态分布方差，刚学过， std是方差。
        # torch.randn 生成正态分布
        Latent_Representation = torch.randn(std.size()) * std + H_2_mean
        return Latent_Representation

    # 解码隐变量
    def Graph_Decoder(self, Latent_Representation):
        graph_re = Graph_Construction(Latent_Representation)
        Graph_Reconstruction = graph_re.Middle()
        return Graph_Reconstruction

    def forward(self, Adjacency_Modified, H_0):
        H_2_mean, H_2_std = self.Encoder(Adjacency_Modified, H_0)
        Latent_Representation = self.Reparametrization(H_2_mean, H_2_std)
        Graph_Reconstruction = self.Graph_Decoder(Latent_Representation)
        return Latent_Representation, Graph_Reconstruction, H_2_mean, H_2_std


class Graph_Construction:
    # Input: n * d.
    def __init__(self, X):
        self.X = X

    def Middle(self):
            Inner_product = self.X.mm(self.X.T)
            Graph_middle = torch.sigmoid(Inner_product)
            return Graph_middle



