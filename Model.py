import torch
from torch import nn
from torch.distributions import Normal
from torch.nn.functional import softplus
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import math
from embedding import DataEmbedding
from torch.autograd import Variable
from layers.Embed import DataEmbedding, DataEmbedding_wo_pos
from Graph import returnA
from utils import GCN_Loss
from Enconder import AnomalyTransformer
from solver import my_kl_loss
from dc import DCdetector
from layers.SelfAttention_Family import Attention_Net,BiFPN_Concat3,AttentionLayer
from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))#权重矩阵
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))#偏移向量
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        input = input.cuda()
        support = torch.mm(input, self.weight)
        adj = adj.clone().detach().cuda()
        support = support.clone().detach().cuda()
        # adj = torch.tensor(adj, dtype=torch.float64)# torch.mm(a, b)是矩阵a和b矩阵相乘，torch.mul(a, b)是矩阵a和b对应位相乘，a和b的维度必须相等此处主要定义的是本层的前向传播，通常采用的是 A*X*W的计算方法。由于A是一个sparse变量，因此其与X进行卷积的结果也是稀疏矩阵。
        # support = torch.tensor(support, dtype=torch.float64)
        output = torch.spmm(adj.double(), support.double())# torch.spmm(a,b)是稀疏矩阵相乘
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCN(nn.Module):
    def __init__(self, nfeat, out):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, out)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        return x

class my_Layernorm(nn.Module):
    """
    Special designed layernorm for the seasonal part
    """

    def __init__(self, channels):
        super(my_Layernorm, self).__init__()
        self.layernorm = nn.LayerNorm(channels)

    def forward(self, x):
        x_hat = self.layernorm(x)
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        return x_hat - bias
class att_encoder(nn.Module):
    """encoder in DA_RNN."""

    def __init__(self,
                 input_size,
                 encoder_num_hidden,
                 batch_size,
                 window_length,
                 parallel=False):
        """Initialize an encoder in DA_RNN."""
        super(att_encoder, self).__init__()
        self.encoder_num_hidden = encoder_num_hidden
        self.input_size = input_size
        self.batch_size = batch_size
        self.window_length = window_length
        self.parallel = parallel
        self.dropout = 0.0
        self.output_attention = True
        self.embedding = DataEmbedding_wo_pos(55, 128, 'timeF', 'h',
                                                  0.1)
        self.auto_attn = Encoder(
            [
                EncoderLayer(
                    Attention_Net(AttentionLayer(
                        AutoCorrelation(False, 1, attention_dropout=0.1,
                                        output_attention=False),
                        128, 8))

                    ,
                    128,
                    128,
                    moving_avg=25,
                    dropout=0.1,
                    activation='glue'
                ) for l in range(3)
            ],
            norm_layer=my_Layernorm(128)
        )
        self.pro = nn.Linear(128,55)
        self.h_n = torch.nn.Parameter(torch.FloatTensor(1, batch_size, encoder_num_hidden), requires_grad=False)
        self.s_n = torch.nn.Parameter(torch.FloatTensor(1, batch_size, encoder_num_hidden), requires_grad=False)

        torch.nn.init.uniform_(self.h_n, a=0, b=0)
        torch.nn.init.uniform_(self.s_n, a=0, b=0)

        self.encoder_lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.encoder_num_hidden,
            num_layers=1
        )

        self.encoder_attn = nn.Linear(
            in_features=2 * self.encoder_num_hidden + self.window_length,
            out_features=1
        )

    def forward(self, X):
        """forward.

        Args:
            X: input data

        """
        # X = X.cuda()
        # enc_out = self.embedding(X,None)
        # enc_out,attns =self.auto_attn(enc_out,attn_mask=None)
        # X = self.pro(enc_out)
        #LSTM注意力
        X_tilde = Variable(X.data.new(
            X.size(0), self.window_length, self.input_size).zero_())

        # batch_size * input_size * (2 * hidden_size + T - 1)
        x = torch.cat((self.h_n.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                        self.s_n.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                        X.permute(0, 2, 1)), dim=2)
        # x = x.cpu()
        x = self.encoder_attn(
            x.view(-1, self.encoder_num_hidden * 2 + self.window_length))


        # get weights by softmax
        alpha = F.softmax(x.view(-1, self.input_size), dim=1)

        for t in range(self.window_length):
            x_tilde = torch.mul(alpha, X[:, t, :])
            X_tilde[:, t, :] = x_tilde

        self.encoder_lstm.flatten_parameters()
        _, final_state = self.encoder_lstm(x_tilde.unsqueeze(0), (self.h_n, self.s_n))
        self.h_n = Parameter(final_state[0])
        self.s_n = Parameter(final_state[1])
        X_tilde = X_tilde.view(-1, self.window_length * self.input_size)

        return X_tilde


def tabular_encoder(input_size: int, latent_size: int):
    """
    Simple encoder for tabular data.
    If you want to feed image to a VAE make another encoder function with Conv2d instead of Linear layers.
    :param input_size: number of input variables
    :param latent_size: number of output variables i.e. the size of the latent space since it's the encoder of a VAE
    :return: The untrained encoder model
    """
    return nn.Sequential(
        nn.Linear(input_size, latent_size * 2),
        # times 2 because this is the concatenated vector of latent mean and variance
    )


def tabular_decoder(latent_size: int, output_size: int):
    """
    Simple decoder for tabular data.
    :param latent_size: size of input latent space
    :param output_size: number of output parameters. Must have the same value of input_size
    :return: the untrained decoder
    """
    return nn.Sequential(
        nn.Linear(latent_size, output_size * 2),

        # times 2 because this is the concatenated vector of reconstructed mean and variance
    )

# from Mamba import MambaBlock
class MUTANT(nn.Module):

    def __init__(self, input_size: int, w_size, hidden_size: int, latent_size, batch_size, window_length, out_dim):
        """
        :param input_size: Number of input features
        :param latent_size: Size of the latent space
        :param L: Number of samples in the latent space (See paper for more details)
        """
        super().__init__()
        self.input_size = input_size
        self.w_size = w_size
        self.latent_size = latent_size
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.att_encoder = att_encoder(input_size, hidden_size, batch_size, out_dim)
        self.att_encoder_ori = att_encoder(input_size, hidden_size, batch_size, out_dim)
        self.encoder = tabular_encoder(w_size, latent_size)
        self.decoder = tabular_decoder(latent_size, w_size)
        self.GCN = GCN(window_length, out_dim)
        self.prior = Normal(0, 1)
        self.d_model = 128
        # self.s1 = nn.Parameter(torch.Tensor([1.0]))
        # self.s2 = nn.Parameter(torch.Tensor([1.0]))
        # self.mamba = MambaBlock(seq_len=20,d_model=55,state_size=128,device='cuda')
        self.win_size = window_length
        self.embedding = DataEmbedding(self.input_size, self.d_model, dropout=0.0)
        self.anomaly_trans = AnomalyTransformer(window_length ,d_model=128, n_heads=8, e_layers=3, d_ff=128,
                 dropout=0.0, activation='gelu', output_attention=True)
        self.projection = nn.Linear(self.d_model, 55, bias=True)
        # self.projection = nn.Linear(self.d_model, input_size, bias=True)
        self.dc = DCdetector(window_length, enc_in=input_size, c_out = out_dim, n_heads=1, d_model=256, e_layers=3, patch_size=[3,6], channel=55, d_ff=512, dropout=0.0, activation='gelu', output_attention=True)
        self.patch_size = [4,5]
        self.auto_attn = Encoder(
            [
                EncoderLayer(
                    Attention_Net(AttentionLayer(
                        AutoCorrelation(False, 1, attention_dropout=0.1,
                                        output_attention=False),
                        128, 8))

                    ,
                    128,
                    128,
                    moving_avg=25,
                    dropout=0.1,
                    activation='glue'
                ) for l in range(3)
            ],
            norm_layer=my_Layernorm(128)
        )
        self.pro = nn.Linear(128,input_size)
    def forward(self, x):
        Xt = []
        l0 = 0
        for i in x:
            A = torch.tensor(returnA(i)).cuda()
            x_g = self.GCN(i.permute(1, 0), A).permute(1, 0).cpu()
            loss = GCN_Loss(x_g)
            l0 = loss
            x_g = x_g.detach().numpy()
            Xt.append(x_g)
        Xt = torch.tensor(Xt, dtype=torch.float32)#   batch_size, window_length, input_size
        Xt = Xt.cuda()
        x_ori = x.cuda()

        # lstm_ori = self.att_encoder_ori(x_ori)
        # lstm_gnn = self.att_encoder(Xt)
        #消融实验 start
        x = Xt
        enc_out = self.embedding(Xt,None)
        enc_out,attns =self.auto_attn(enc_out,attn_mask=None)
        X_w = self.pro(enc_out)
        X_w = X_w.view(-1,X_w.shape[1]*X_w.shape[2])
        # 消融实验 end
        lstm_out = self.att_encoder(x_ori)
        # X_w  = Xt.view(-1,Xt.shape[1]*Xt.shape[2])
        # X_w = X_wlstm_out
        # 消融实验 start
        X_w = X_w+0*lstm_out
        # 消融实验 end
        x = Xt

        # Xt = torch.tensor(Xt, dtype=torch.float32)  # batch_size, window_length, input_size


        # Xt = self.embedding(Xt)
        # enc_out, series, prior, sigmas = self.anomaly_trans(Xt)
        # X_w = enc_out.view(120,-1)
        # X_w = self.projection(enc_out)
        # mamba = self.mamba(x)
        # X_w = X_w+mamba
        # X_w = self.att_encoder(Xt)
        # series = [tensor.cuda() for tensor in series]
        # enc_out = self.projection(enc_out)
        # series_loss = 0.0
        # prior_loss = 0.0
        # for u in range(len(prior)):
        #     series_loss += (torch.mean(my_kl_loss(series[u], (
        #             prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
        #                                                                                    self.win_size)).detach())) + torch.mean(
        #         my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
        #                                                                                            self.win_size)).detach(),
        #                    series[u])))
        #     prior_loss += (torch.mean(my_kl_loss(
        #         (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
        #                                                                                 self.win_size)),
        #         series[u].detach())) + torch.mean(
        #         my_kl_loss(series[u].detach(), (
        #                 prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
        #                                                                                        self.win_size)))))
        # series_loss = series_loss / len(prior)
        # prior_loss = prior_loss / len(prior)
        pred_result = self.predict(X_w)
        mu = pred_result['recon_mu']
        Xt = x.view(-1, x.shape[1] * x.shape[2])
        loss_function = torch.nn.MSELoss()
        l1 = loss_function(mu, Xt)
        l2 = torch.mean(-0.5 * torch.sum(1 + pred_result['latent_sigma'] - pred_result['latent_mu'].pow(2) - pred_result['latent_sigma'].exp(), dim=1), dim=0)
        l0 = l1+l0+l2
        return l0

    def predict(self, x) -> dict:
        """
        :param x: tensor of shape [batch_size, num_features]
        :return: A dictionary containing prediction i.e.
        - latent_dist = torch.distributions.Normal instance of latent space
        - latent_mu = torch.Tensor mu (mean) parameter of latent Normal distribution
        - latent_sigma = torch.Tensor sigma (std) parameter of latent Normal distribution
        - recon_mu = torch.Tensor mu (mean) parameter of reconstructed Normal distribution
        - recon_sigma = torch.Tensor sigma (std) parameter of reconstructed Normal distribution
        - z = torch.Tensor sampled latent space from latent distribution
        """
        # x = x.cpu()
        batch_size = len(x)
        latent_mu, latent_sigma = self.encoder(x).chunk(2, dim=1)
        latent_sigma = softplus(latent_sigma) + 1e-4
        dist = Normal(latent_mu, latent_sigma)
        z = dist.rsample()
        z = z.view(batch_size, self.latent_size)
        recon_mu, recon_sigma = self.decoder(z).chunk(2, dim=1)
        recon_sigma = softplus(recon_sigma) + 1e-4
        recon_mu = recon_mu.view(-1, x.shape[1])
        recon_sigma = recon_sigma.view(-1, x.shape[1])

        return dict(latent_dist=dist, latent_mu=latent_mu,
                latent_sigma=latent_sigma, recon_mu=recon_mu,
                recon_sigma=recon_sigma, z=z)


    def is_anomaly(self, x, num_t, con_t):
        """

        :param x:
        :param alpha: Anomaly threshold (see paper for more details)
        :return: Return a vector of boolean with shape [x.shape[0]]
                 which is true when an element is considered an anomaly
        """
        score = []

        np.set_printoptions(threshold=999999999999999999)
        for i, inputs in enumerate(x, 0):
            p = self.reconstructed_probability(inputs)
            if i==num_t:
                p = p[:con_t]
            score = np.concatenate((score, p), axis=0)
        # score = [torch.from_numpy(array).cpu() for array in score]
        score = np.array(score)
        return score

    def reconstructed_probability(self, x):
        with torch.no_grad():
            Xt = []
            for i in x:
                A = torch.tensor(returnA(i))
                x_g = self.GCN(i.permute(1, 0), A).permute(1, 0).detach().cpu()
                x_g = x_g.numpy()
                Xt.append(x_g)
            x_ori = x.cuda()
            Xt = torch.tensor(Xt, dtype=torch.float32).cuda()
            xt = Xt
            # Xt = self.embedding(Xt)
            # enc_out, series, prior, sigmas = self.anomaly_trans(Xt)

            # enc_out, series, prior, sigmas = self.anomaly_trans(Xt)
            # X_w = self.projection(enc_out)
            # mamba = self.mamba(xt)
            # X_w = X_w + mamba
            # X_w = self.att_encoder(Xt)
            #消融实验start
            enc_out = self.embedding(Xt, None)
            enc_out, attns = self.auto_attn(enc_out, attn_mask=None)
            X_w = self.pro(enc_out)
            X_w = X_w.view(-1, X_w.shape[1] * X_w.shape[2])
            #消融实验end
            lstm_out = self.att_encoder(x_ori)
            # X_w = Xt.view(-1, Xt.shape[1] * Xt.shape[2])
            # X_w = X_w + lstm_out
            # X_w = lstm_out
            #消融实验start
            X_w = X_w + 0*lstm_out
            #消融实验end
            # pred = self.projection(enc_out)
            # mu = pred['recon_mu']
            pred = self.predict(X_w)
            mu = pred['recon_mu']
            Xt = xt.view(-1, xt.shape[1] * xt.shape[2])
            p = []
            for i in range(x.shape[0]):
                t = abs(torch.sum((Xt[i] - mu[i])))
                p.append(t)
        p = [list.cpu() for list in p]
        p = np.array(p)
        return p

    def generate(self, batch_size: int=1) -> torch.Tensor:
        """
        Sample from prior distribution, feed into decoder and get in output recostructed samples
        :param batch_size:
        :return: Generated samples
        """
        z = self.prior.sample((batch_size, self.latent_size))
        recon_mu, recon_sigma = self.decoder(z).chunk(2, dim=1)
        recon_sigma = softplus(recon_sigma)
        return recon_mu + recon_sigma * torch.rand_like(recon_sigma)


