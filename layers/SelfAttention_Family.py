import torch
import torch.nn as nn
import numpy as np
from math import sqrt
from layers.masking import TriangularCausalMask, ProbMask
from layers.TCN import TemporalConvNet
from layers.Atten_SCINet import SCINet




class DSAttention(nn.Module):
    '''De-stationary Attention'''

    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(DSAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape#B:128 E:16 L:100 H:8
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        tau = 1.0 if tau is None else tau.unsqueeze(
            1).unsqueeze(1)  # B x 1 x 1 x 1
        delta = 0.0 if delta is None else delta.unsqueeze(
            1).unsqueeze(1)  # B x 1 x 1 x S

        # De-stationary Attention, rescaling pre-softmax score with learned de-stationary factors
        scores = torch.einsum("blhe,bshe->bhls", queries, keys) * tau + delta #scores(128,8,100,100)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1)) #A(128,8,100,100)
        V = torch.einsum("bhls,bshd->blhd", A, values) #V(128,100,8,16)
        V_C = V.contiguous()
        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)




class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
        # self.con = nn.Conv1d(num_channels[-1],output_size,)
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        y1 = self.tcn(x)
        # y1.permute(0, 2, 1)
        # y1 = self.linear(y1)
        return y1.permute(0, 2, 1)

class ParallelDSConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=2, dilation=1):
        super(ParallelDSConv, self).__init__()
        self.depthwise_conv = nn.Conv1d(in_channels, in_channels, kernel_size, stride, padding, dilation,
                                        groups=in_channels)
        self.pointwise_conv = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        x = x.permute(0, 2, 1)
        return x


class ParallelDilatedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=4, dilation=1):
        super(ParallelDilatedConv, self).__init__()
        self.dilated_conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation)
        self.conv = nn.Conv1d(96,100,1)


    def forward(self, x):
        x = self.dilated_conv(x)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        return x


# class ParallelModel(nn.Module):
#     def __init__(self):
#         super(ParallelModel, self).__init__()
#         self.dsattention = DSAttention()
#         self.dsconv = ParallelDSConv(in_channels=128, out_channels=128, kernel_size=5)
#         self.dilatedconv = ParallelDilatedConv(in_channels=128, out_channels=128, kernel_size=7, dilation=2)
#
#     def forward(self, queries, keys, values, attn_mask):
#         attn_out, attn = self.dsattention(queries, keys, values, attn_mask)
#         dsconv_out = self.dsconv(queries)
#         dilatedconv_out = self.dilatedconv(queries)
#         out = attn_out + dsconv_out + dilatedconv_out
#
#         return out






class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        # self.w_g = nn.Parameter(torch.randn(19200, 16))
        self.scale_factor = d_model ** -0.5
        self.query_Conv = nn.Conv1d(d_model,d_keys*n_heads,1,stride=1)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_Conv = nn.Conv1d(d_model, d_keys * n_heads, 1, stride=1)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_Conv = nn.Conv1d(d_model, d_values * n_heads, 1, stride=1)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.out_Conv = nn.Sequential(nn.Conv2d(d_values * n_heads,d_values * n_heads,7,stride=1,padding=3,
                                                groups=d_values * n_heads),
                                      nn.Conv2d(d_values * n_heads,d_model,1))
        self.n_heads = n_heads
        # self.dsconv = ParallelDSConv(in_channels=128, out_channels=128, kernel_size=5)
        # self.dilatedconv = ParallelDilatedConv(in_channels=128, out_channels=128, kernel_size=7, dilation=2)

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries)
        queries = nn.functional.normalize(queries, dim=-1)
        # queries = queries.permute(0, 2, 1)
        # dsconv_out = self.dsconv(queries)
        # dilatedconv_out = self.dilatedconv(queries)
        # queries = queries.permute(0, 2, 1)
        queries = queries.view(B, S, H, -1)
        keys = self.key_Conv(keys.transpose(1,-1)).transpose(1,-1).view(B, S, H, -1)
        keys = nn.functional.normalize(keys, dim=-1)
        # query_weight = queries @ self.w_g
        values = self.value_Conv(values.transpose(1,-1)).transpose(1,-1).view(B, S, H, -1)

        attn_out, attn = self.inner_attention(   #out (128,100,8,16)
            queries,
            keys,
            values,
            attn_mask,
        )


        out = attn_out.view(B,L,-1) # out (128,100,128)
        # out = self.out_Conv(out).transpose(1,-1).squeeze(1)

        return self.out_projection(out), attn

from layers.cad import MMoE,LKA

class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        # real U = U_part(factor*ln(L_k))*L_q
        index_sample = torch.randint(L_K, (L_Q, sample_k))
        K_sample = K_expand[:, :, torch.arange(
            L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(
            Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H,
                                                L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            # requires that L_Q == L_V, i.e. for self-attention only
            assert (L_Q == L_V)
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) /
                     L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[
                                                  None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * \
                 np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * \
            np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(
            queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(
            context, values, scores_top, index, L_Q, attn_mask)

        return context.contiguous(), attn

class BiFPN_Concat3(nn.Module):
    def __init__(self, dimension=1):
        super(BiFPN_Concat3, self).__init__()
        self.d = 2
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001

    def forward(self, x):
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)  # 将权重进行归一化
        # Fast normalized fusion
        x = weight[0] * x[0]+weight[1] * x[1]
        return x

# from layers.simAM import TimeSeriesModel
class Attention_Net(nn.Module):
    def __init__(self, attentionlayer):
        super(Attention_Net, self).__init__()
        num_channels = [64, 128, 256, 128]  # 每个 TemporalBlock 的通道数，可以根据需求调整
        self.pm1 = SCINet(output_len=20, input_len=20, input_dim=128, hid_size=1,
                   num_stacks=1,
                   num_levels=2, concat_len=0, groups=1, kernel=5, dropout=0.1,
                   single_step_output_One=0, positionalE=False,
                   modified=True)

        self.pm2 = attentionlayer
        self.mmoe = MMoE()
        self.mmoe1 = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.Concat3 = BiFPN_Concat3()

    def forward(self, x, queries, keys, values, attn_mask, tau=None, delta=None):
        sp1, attn = self.pm2(queries, keys, values, attn_mask)
        sp1_1 =sp1
        sp1_1 = self.relu(sp1_1)
        sp1 = self.mmoe(sp1)
        # sp1 = self.sigmoid(sp1)
        sp1 = sp1*sp1_1
        # sp2, attn = self.pm2(queries, keys, values, attn_mask)
        sp2 = self.pm1(x)
        # sp2_2, attn2= self.pm2(queries, keys, values, attn_mask)

        # sp2 = self.relu(sp2)
        # attn1 = self.sigmoid(attn1)
        # attn2 = self.relu(attn2)


        out = self.Concat3([sp1,sp2])


        # out = sp1+sp2
        # attn = attn1*attn2
        return sp2, attn


class ReformerLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None, causal=False, bucket_size=4, n_hashes=4):
        super().__init__()
        self.bucket_size = bucket_size

    def fit_length(self, queries):
        # inside reformer: assert N % (bucket_size * 2) == 0
        B, N, C = queries.shape
        if N % (self.bucket_size * 2) == 0:
            return queries
        else:
            # fill the time series
            fill_len = (self.bucket_size * 2) - (N % (self.bucket_size * 2))
            return torch.cat([queries, torch.zeros([B, fill_len, C]).to(queries.device)], dim=1)

    def forward(self, queries, keys, values, attn_mask, tau, delta):
        # in Reformer: defalut queries=keys
        B, N, C = queries.shape
        queries = self.attn(self.fit_length(queries))[:, :N, :]
        return queries, None


