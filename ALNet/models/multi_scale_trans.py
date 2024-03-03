import os
import sys
sys.path.append(".")
import math
import random
# import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import Module, Dropout
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from .resnet_backbone import resnet18
from .utils import PositionEncodingSine
import matplotlib.pyplot as plt
import numpy as np

#transformer相关模块
def elu_feature_map(x):
    return torch.nn.functional.elu(x) + 1 #elu激活函数

#222
#线性注意力
class LinearAttention(Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.feature_map = elu_feature_map
        self.eps = eps

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        """ Multi-Head linear attention proposed in "Transformers are RNNs"
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """
        Q = self.feature_map(queries)
        K = self.feature_map(keys)

        # set padded position to zero
        if q_mask is not None:
            Q = Q * q_mask[:, :, None, None]
        if kv_mask is not None:
            K = K * kv_mask[:, :, None, None]
            values = values * kv_mask[:, :, None, None]
        # values: [N, S, H, D]
        v_length = values.size(1)
        values = values / v_length  # prevent fp16 overflow
        KV = torch.einsum("nshd,nshv->nhdv", K, values)  # (S,D)' @ S,V
        Z = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps)
        queried_values = torch.einsum("nlhd,nhdv,nlh->nlhv", Q, KV, Z) * v_length

        return queried_values.contiguous() #返回一个连续存储张量 提升操作性能

class ltransformer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                ):#linear自注意力
        super(ltransformer, self).__init__()

        self.dim = d_model // nhead #d model就是nlc中的c
        self.nhead = nhead

        # multi-head attention 多头注意力
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention = LinearAttention() 
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model*2, d_model*2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model*2, d_model, bias=False),
        )
        
        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    ####
    def forward(self, x, source, x_mask=None, source_mask=None):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
        # #suppose x and source are n c h w
        # N1,C1,H1,W1 = x.shape()
        # x = x.reshape(N1, C1, -1)
        # x = x.permute(0, 2, 1)
        # N2,C2,H2,W2 = source.shape()
        # source = source.reshape(N2, C2, -1)
        # source = source.permute(0, 2, 1)

        bs = x.size(0)
        # N, L, C = x.size()
        query, key, value = x, source, source

        # multi-head attention
        query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)
        message = self.attention(query, key, value, q_mask=x_mask, kv_mask=source_mask)  # [N, L, (H, D)]
        message = self.merge(message.view(bs, -1, self.nhead*self.dim))  # [N, L, C]
        message = self.norm1(message)

        # feed-forward network
        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.norm2(message)

        return x + message

class multrans(nn.Module):
    def __init__(self, cin, cout):
        super(self, multrans).__init__()
        #
        self.conv1 = conv(cin, cin, stride=4, padding=0, kernel_size=4)
        self.mlp1 = nn.Sequential(
            nn.Linear(cin, cout, bias=False),
            nn.ReLU(True),
            nn.Linear(cout, cout, bias=False),
        )
        #
        self.conv2 = conv(cin, cin, stride=8, padding=0, kernel_size=8)
        self.mlp2 = nn.Sequential(
            nn.Linear(cin, cout, bias=False),
            nn.ReLU(True),
            nn.Linear(cout, cout, bias=False),
        )
        #
        self.conv3 = conv(cin, cin, stride=16, padding=0, kernel_size=16)
        self.mlp3 = nn.Sequential(
            nn.Linear(cin, cout, bias=False),
            nn.ReLU(True),
            nn.Linear(cout, cout, bias=False),
        )
        #
        self.cout = cout 
        self.ltrans23 = ltransformer(d_model=cout, nhead=4) ##??
        self.ltrans12 = ltransformer(d_model=cout, nhead=4) 

    def forward(self, x):
        #x n c h w 
        n,c,h,w = x.shape
        #
        x1 = self.conv1(x)
        n1,c1,h1,w1 = x1.shape
        x1 = x1.reshape(n,c,-1).permute(0,2,1) #n l1 c
        x1 = self.mlp1(x1)
        #
        x2 = self.conv2(x)
        x2 = x2.reshape(n,c,-1).permute(0,2,1) #n l2 c
        x2 = self.mlp2(x2)
        #
        x3 = self.conv3(x)
        x3 = x3.reshape(n,c,-1).permute(0,2,1) #n l3 c
        x3 = self.mlp3(x3)
        #x2 x3---q kv
        x23 = self.ltrans23(x2, x3)
        print(x23.shape)
        #x1 x2 ---q kv
        out = self.ltrans12(x1, x23)
        #
        out = out.permute(0,2,1).reshape(n1,c1,h1,w1) 
        print(out.shape)
        return out 
#222

##卷积+bn+relu模块
def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                  stride=stride, padding=padding),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True)
    )

#senet模块
class SENet(nn.Module):
    def __init__(self, in_chan, out_chan, stride=1, padding=1, kernel_size=3):
        super(SENet, self).__init__()
        self.conv = conv(in_chan, out_chan, stride=stride, padding=padding, kernel_size=kernel_size)
        self.conv_atten = nn.Conv2d(out_chan, out_chan, kernel_size=1, bias=False)
        self.bn_atten = nn.BatchNorm2d(out_chan)
        self.sigmoid_atten = nn.Sigmoid()

    def forward(self, x):
        feat = self.conv(x)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = self.sigmoid_atten(atten)
        out = torch.mul(feat, atten)
        return out

#######相似度消融
#欧氏距离 最大为1，最小为0
def compute_euclidean_similarity(batch):
    """
    计算一个batch内部的点之间的相似度矩阵
    :param batch: 输入数组，形状为(batch_size, num_channels, num_points)
    :return: 相似度矩阵，形状为(batch_size, num_points, num_points)
    """
    dists = euclidean_distance(batch)
    sim = 1 / (1 + dists)  # 取欧式距离的倒数得到相似度
    return sim
def euclidean_distance(batch):
    """
    计算一个batch内部的点之间的欧式距离
    :param batch: 输入数组，形状为(batch_size, num_channels, num_points)
    :return: 欧式距离矩阵，形状为(batch_size, num_points, num_points)
    """
    batch_size, num_channels, num_points = batch.shape
    batch_reshaped = batch.reshape(batch_size, -1, num_channels)  # 将每个batch展平为(batch_size, num_points, num_channels)
    dists = torch.sqrt(torch.sum((batch_reshaped[:, :, np.newaxis, :] - batch_reshaped[:, np.newaxis, :, :]) ** 2, axis=-1))
    return dists

##曼哈顿距离 最大为1 设置0
def compute_manhattan_similarity(batch):
    """
    计算一个batch内部的点之间的相似度矩阵
    :param batch: 输入数组，形状为(batch_size, num_channels, num_points)
    :return: 相似度矩阵，形状为(batch_size, num_points, num_points)
    """
    dists = manhattan_distance(batch)
    sim = 1 / (1 + dists)  # 取欧式距离的倒数得到相似度
    return sim

def manhattan_distance(batch):
    """
    计算一个batch内部的点之间的曼哈顿距离作为相似度
    :param batch: 输入数组，形状为(batch_size, num_channels, num_points)
    :return: 曼哈顿相似度矩阵，形状为(batch_size, num_points, num_points)
    """
    batch_size, num_channels, num_points = batch.shape
    batch_reshaped = batch.reshape(batch_size, num_channels, -1)  # 将每个batch展平为(batch_size, num_channels, num_points)
    # 计算曼哈顿距离
    distances = torch.abs(batch_reshaped[:, :, :, np.newaxis] - batch_reshaped[:, :, np.newaxis, :])
    manhattan_distances = torch.sum(distances, axis=1)  # 沿着通道维度求和，得到曼哈顿距离
    return manhattan_distances

#余弦相似度 0到1 设置为1
def compute_cosine_similarity(batch):
    """
    计算一个batch内部的点之间的余弦相似度
    :param batch: 输入数组，形状为(batch_size, num_channels, num_points)
    :return: 余弦相似度矩阵，形状为(batch_size, num_points, num_points)
    """
    batch_size, num_channels, num_points = batch.shape
    batch_reshaped = batch.reshape(batch_size, -1, num_channels)  #
    # 计算每个点的模
    norms = torch.linalg.norm(batch_reshaped, axis=-1, keepdims=True)
    # 计算点积
    dot_products = torch.einsum('ijk,ihk->ijh', batch_reshaped, batch_reshaped)
    # 计算余弦相似度
    similarities = dot_products / (norms * torch.transpose(norms, 1, 2))
    return similarities


#新增 去相似化模块 输入为 4 512 480/8=60 640/8=80 n c h w
class Desimilar_block(nn.Module):
    def __init__(self, exp_size, top_number, c, down_sample, up_sample):
        super(Desimilar_block, self).__init__()
        self.conv = SENet(2*c, c, stride=1, padding=1, kernel_size=3) 
        ##
        # self.conv_se = SENet(c, c, stride=1, padding=0, kernel_size=1) 
        # self.se = se
        ##
        self.conv_down = conv(c, c, stride=down_sample, padding=0, kernel_size=down_sample)
        # self.conv_down = conv(512, 512, stride=2, padding=1, kernel_size=3)
        self.exp_size = exp_size
        self.top_number = top_number
        self.up_sample = up_sample
    ###
    def make_i(self, i, exp_size):
        win_size = 2*exp_size+1
        t = torch.ones((win_size,win_size))
        # print(t.shape)
        for m in range(0,win_size):
            t[m,:] = i-exp_size+m
        t = t[None,:,:].repeat(1,1,1).reshape(1,win_size*win_size)
        return t #1 25
    ###
    def make_j(self, j, exp_size):
        win_size = 2*exp_size+1
        t = torch.ones((win_size,win_size))
        for m in range(0,win_size):
            t[:,m] = j-exp_size+m
        t = t[None,:,:].repeat(1,1,1).reshape(1,win_size*win_size)
        return t #1 25
    ###
    def get_feat(self, feat, exp_size, top_number):
        #获取特征图维度
        B,C,H,W = feat.shape
        L = H*W
        #窗口扩展大小
        # exp_size = 2
        window_size = 2*exp_size+1
        #选择相似度最低的个数
        # top_number = 9
        #特征图维度变为 4，512，4800
        x1 = feat.reshape(B,C,L)
        #矩阵相乘求解相似度矩阵 4，4800，4800
        similarity_matrix = torch.einsum('nci,ncj->nij', x1, x1)
        
        #####相似度消融
        #欧式距离 最大为1 最小为0
        similarity_matrix = compute_euclidean_similarity(x1)
        #曼哈顿距离 最大为1， 最小为0
        #similarity_matrix = compute_manhattan_similarity(x1)
        #余弦相似度
        #similarity_matrix = compute_cosine_similarity(x1)
        # print('ok')
        # exit()
        #####

        # import matplotlib.pyplot as plt
        # plt.imshow(similarity_matrix[0].detach().cpu().numpy())
        # plt.show()
        # exit()
        #维度变为4*4800*60*80 4*1200*30*40 
        similarity_matrix = similarity_matrix.reshape(B,L,H,W)
        # import matplotlib.pyplot as plt
        # plt.imshow(similarity_matrix[0][900].detach().cpu().numpy())
        # plt.show()
        # exit()
        #相似度矩阵周围填充两圈0 零填充操作 
        padding_size = (exp_size,exp_size,exp_size,exp_size) #四个参数代表对最后两个维度扩充
        ##维度变为4*4800*64*84  原始设成1e20 
        pad_simi_matrix = F.pad(similarity_matrix, padding_size,'constant',1e20)
        # import matplotlib.pyplot as plt
        # plt.imshow(torch.sum(similarity_matrix[0], dim=0).detach().cpu().numpy())
        # plt.show()

        # import matplotlib.pyplot as plt
        # plt.imshow(similarity_matrix[0][900].detach().cpu().numpy())
        # plt.show()
        # exit()

        #获取25的索引4 4800 25并在相似度矩阵上索引
        b_index = torch.arange(B)[:,None,None].repeat(1,L,window_size**2)
        l_index = torch.arange(L)[None,:,None].repeat(B,1,window_size**2)
        I = torch.cat([self.make_i(i, exp_size) for i in range(exp_size,H+exp_size)],dim=0)
        i_index = I[None,:,None,:].repeat(B,1,W,1).reshape(B,L,window_size**2)
        J = torch.cat([self.make_j(j, exp_size) for j in range(exp_size,W+exp_size)],dim=0)
        j_index = J[None,None,:,:].repeat(B,H,1,1).reshape(B,L,window_size**2)
        #获取4*4800*25的相似度矩阵上的值
        win_value = pad_simi_matrix[b_index.long(),l_index.long(),i_index.long(),j_index.long()]
        #获取topk索引 4*4800*9 值为0-24
        top_values, top_indices = torch.topk(win_value,k=top_number,dim=2,largest=False)
        #获取h w方向的偏移量
        h_bias = top_indices // window_size-exp_size
        w_bias = top_indices %  window_size-exp_size
        #获取索引 4*4800*9
        center = torch.arange(L)[None, :, None].repeat(B, 1, top_number).to(h_bias.device)
        # h_bias = h_bias.to(center.device)
        # w_bias = w_bias.to(center.device) ##cpu
        H_f = center // W + h_bias
        W_f = center % W + w_bias
        index = H_f * W + W_f
        #处理特征图 x1：n c l - n l c
        x_stron = x1.permute(0,2,1)
        x_stron = x_stron[:,:,None,:].repeat(1,1,L,1)
        a_index = torch.arange(B)[:,None,None].repeat(1,L,top_number)
        b_index = torch.arange(L)[None,:,None].repeat(B,1,top_number)
        x_final = x_stron[a_index.long(),b_index.long(),index.long(),:] ##4*4800*9*c
        return x_final
    ###用卷积的方法
    def forward(self, x): ##x: n c h w 
        x1 = self.conv_down(x)
        # import matplotlib.pyplot as plt
        # plt.imshow(torch.sum(x1[0], dim=0).detach().cpu().numpy())
        # plt.show()
        # exit()
        ##
        # if self.se is not None:
        #     x1 = self.conv_se(x1)
        ##
        B,C,H,W = x1.shape
        L = H*W
        out_1 = self.get_feat(x1, self.exp_size, self.top_number)# n l 9 c
        out_0 = x1.reshape(B,C,L).permute(0,2,1) #n,l,c
        out_0 = out_0[:,:,None,:].repeat(1,1,1,1) #n l 1 c
        out = out_1 - out_0#通过广播机制求差异 4*4800*9*c
        out = torch.mean(out, dim=2, keepdim=False) #4*4800*C
        out = out.permute(0,2,1).reshape(B,-1,H,W)#B C H W

        # import matplotlib.pyplot as plt
        # plt.imshow(torch.sum(x1[0], dim=0).detach().cpu().numpy())
        # plt.show()
        # exit()
        out = torch.cat((x1, out), dim=1)#B 2C H W
        out = self.conv(out)

        # import matplotlib.pyplot as plt
        # plt.imshow(torch.sum(out[0], dim=0).detach().cpu().numpy())
        # plt.show()
        # exit()
        out = F.interpolate(out, size=[H*self.up_sample,W*self.up_sample], mode='bilinear', align_corners=False)
        print('niubiniubi')
        return out

##新增 通道注意力模块 n c h w 两个参数为输入c和输出c
class att_layer(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(att_layer, self).__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.global_maxpool = nn.AdaptiveMaxPool2d(1)
        self.conv_1 = nn.Conv2d(in_chan*3, in_chan, kernel_size=1, stride=1, padding=0)
        self.conv_2 = nn.Conv2d(in_chan, in_chan, kernel_size=1,stride=1, padding=0)
        self.conv_3 = conv(in_chan, out_chan, stride=1, padding=0, kernel_size=1)
        self.bn_atten_1 = nn.BatchNorm2d(in_chan)
        self.bn_atten_2 = nn.BatchNorm2d(in_chan)
        self.sigmoid_atten_1 = nn.Sigmoid()
        self.sigmoid_atten_2 = nn.Sigmoid()
    def forward(self, x):
        out_1 = self.global_avgpool(x)
        out_2 = self.global_maxpool(x)
        out = out_2 - out_1
        out = torch.cat([out, out_1, out_2], dim=1)
        out_w1 = self.conv_1(out)
        out_w1 = self.bn_atten_1(out_w1)
        out_w1 = self.sigmoid_atten_1(out_w1)
        out_w2 = 1 - out_w1
        out_1 = out_1 * out_w1
        out_2 = out_2 * out_w2
        out = out_1 + out_2
        out = self.conv_2(out)
        out = self.bn_atten_2(out)
        out = self.sigmoid_atten_2(out)
        out = out * x + x
        out = self.conv_3(out)
        return out

class att_layer2(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(att_layer2, self).__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv_2 = nn.Conv2d(in_chan, in_chan, kernel_size=1,stride=1, padding=0)
        self.conv_3 = conv(in_chan, out_chan, stride=1, padding=0, kernel_size=1)
        self.bn_atten_2 = nn.BatchNorm2d(in_chan)
        self.sigmoid_atten_2 = nn.Sigmoid()
    def forward(self, x):
        out1 = self.global_avgpool(x)
        out = self.conv_2(out1)
        out = self.bn_atten_2(out)
        out = self.sigmoid_atten_2(out)
        out = out * x + x
        out = self.conv_3(out)
        print('attn changed')

        return out

#输入：[layer2, layer3, layer4] c=128，256，512
#输出：c=512+512=1024 8倍下采样
class module_1(nn.Module):
    def __init__(self):
        super(module_1, self).__init__()
        #调用通道注意力模块
        self.att_layer = att_layer(in_chan=(512+256+128),out_chan=512)
        #
        self.conv_high = conv(128, 128, stride=2, padding=1, kernel_size=3)
    def forward(self, x):
        #对layer2下采样
        x0, x1, x2 = x[0], x[1], x[2]
        x0 = self.conv_high(x0) 
        out = torch.cat([x0, x1, x2], dim=1) #c=128+256+512
        #通道注意力
        out = self.att_layer(out)
        out = torch.cat([x2, out], dim=1)
        return out

#222
class fusetrans(nn.Module):     ##128-256-512
    def __init__(self, cout):
        super(self, fusetrans).__init__()
        self.SE1 = SENet(in_chan=128, out_chan=cout, stride=2)
        self.SE2 = SENet(in_chan=256, out_chan=cout, stride=1)
        self.SE3 = SENet(in_chan=512, out_chan=cout, stride=1) #60*80*c
        #
        self.cout = cout 
        self.ltrans12 = ltransformer(d_model=cout, nhead=4)
        self.ltrans23 = ltransformer(d_model=cout, nhead=4) ##??
    def forward(self, x1, x2, x3):
        x1 = self.SE1(x1)
        x2 = self.SE2(x2)
        x3 = self.SE3(x3)
        #x n c h w 
        n,c,h,w = x1.shape
        x1 = x1.reshape(n,c,-1).permute(0,2,1) #n l c
        x2 = x2.reshape(n,c,-1).permute(0,2,1) #n l c
        x3 = x3.reshape(n,c,-1).permute(0,2,1) #n l c
        #x2 x1---q kv
        x12 = self.ltrans12(x2, x1)
        print(x12.shape)
        #x3 x12 ---q kv
        out = self.ltrans23(x3, x12)
        #
        out = out.permute(0,2,1).reshape(n,c,h,w) 
        print(out.shape)
        return out 
#222

#新增 总体模块
class Multi_Scale_Trans(nn.Module):
    def __init__(self):
        super(Multi_Scale_Trans, self).__init__()
        cnn = resnet18(pretrained=True)
        self.cnn_pre = nn.Sequential(cnn.conv1, cnn.bn1, cnn.relu)
        self.layer1 = cnn.layer1
        self.layer2 = cnn.layer2
        self.layer3 = cnn.layer3
        self.layer4 = cnn.layer4
        #ccccc
        self.last_conv = nn.Sequential(
            conv(512+512+64, 512),
            conv(512, 256),
            conv(256, 128)
        )###16需要更改
        self.coord_regress = nn.Sequential(
            conv(128, 64),
            nn.Conv2d(64, 3, kernel_size=1, padding=0)
        )
        self.uncer_regress = nn.Sequential(
            conv(128, 64),
            nn.Conv2d(64, 1, kernel_size=1, padding=0)
        )
        #
        B,C,H,W = 4, 64, 240, 320
        self.pos_embed = PositionEncodingSine(C, max_shape=(H, W))
        #新增 去相似化模块 
        self.desimilar_block = Desimilar_block(exp_size=2,top_number=6,c=64,down_sample=8,up_sample=2)
        #新增 融合模块
        # self.fuse_layer_1 = fuse_layer(chan0=128,chan1=256)
        # self.fuse_layer_2 = fuse_layer(chan0=256,chan1=512)
        # self.conv_fuse = conv(512, 512, stride=2, padding=1, kernel_size=3) #cccc
        # 新增 通道注意力模块
        self.att_layer = att_layer(in_chan=(512+256+128),out_chan=512)
        #self.att_layer2 = att_layer2(in_chan=(512+256+128),out_chan=512)
        #
        self.conv_high = conv(128, 128, stride=2, padding=1, kernel_size=3)
        # self.p_att_layer = p_att_layer(chan0=128)

    def forward(self, x):
        ##卷积+位置编码
        out = self.cnn_pre(x)
        out = out + self.pos_embed(out).cuda() ###???
        out_des = out 
        ##主干网络
        out_layer1 = self.layer1(out)#
        out_layer2 = self.layer2(out_layer1)
        out_layer3 = self.layer3(out_layer2)
        out_layer4 = self.layer4(out_layer3)
        ##1 c=512
        out_layer4_1 = out_layer4
        #2 新增 去相似化模块 c=512
        out_layer_des = self.desimilar_block(out_des)
        #2 新增 去相似化模块 c=512
        # out_layer4_2 = self.desimilar_block(out_layer4)
        #3 新增 特征融合模块 c=512
        # inputs_1 = [out_layer2, out_layer3]
        # out_mid = self.fuse_layer_1(inputs_1)
        # input_2 = [out_mid, out_layer4]
        # out_layer4_3 = self.fuse_layer_2(input_2) 
        # out_layer4_3 = self.conv_fuse(out_layer4_3)
        # out_layer4_3 = torch.cat([out_layer4_3, out_layer4], dim=1)
        # out_layer4_3 = self.att_layer(out_layer4_3)
        #
        #下采样2
        out_layer2_2 = self.conv_high(out_layer2) 
        out = torch.cat([out_layer4_1, out_layer2_2, out_layer3], dim=1) #c=512+256+128
        # 通道注意力
        out = self.att_layer(out)
        out = torch.cat([out_layer_des, out_layer4_1, out], dim=1)
        # out = torch.cat([out_layer_des, out_layer4_1], dim=1)
        #回归
        out = self.last_conv(out)
        coord = self.coord_regress(out)
        # print(coord.shape)
        # import matplotlib.pyplot as plt
        # plt.imshow(torch.sum(coord[0], dim=0).detach().cpu().numpy(), cmap="jet")
        # plt.show()
        # exit()
        uncer = self.uncer_regress(out)
        uncer = torch.sigmoid(uncer)
        # plt.imshow(torch.sum(uncer[0], dim=0).detach().cpu().numpy(),cmap="jet")
        # plt.colorbar()
        # plt.show()
        # exit()
        return coord, uncer

#新增 总体模块222
class Multi_Scale_Trans2(nn.Module):
    def __init__(self):
        super(Multi_Scale_Trans2, self).__init__()
        cnn = resnet18(pretrained=True)
        self.cnn_pre = nn.Sequential(cnn.conv1, cnn.bn1, cnn.relu)
        self.layer1 = cnn.layer1
        self.layer2 = cnn.layer2
        self.layer3 = cnn.layer3
        self.layer4 = cnn.layer4
        #ccccc
        self.last_conv = nn.Sequential(
            conv(512+64+256, 512),
            conv(512, 256),
            conv(256, 128)
        )###16需要更改
        self.coord_regress = nn.Sequential(
            conv(128, 64),
            nn.Conv2d(64, 3, kernel_size=1, padding=0)
        )
        self.uncer_regress = nn.Sequential(
            conv(128, 64),
            nn.Conv2d(64, 1, kernel_size=1, padding=0)
        )
        #
        B,C,H,W = 4, 64, 240, 320
        self.pos_embed = PositionEncodingSine(C, max_shape=(H, W))
        #新增 64
        self.multrans = multrans(cin=64, cout=64)
        #新增 128 256 512
        self.fusetrans = fusetrans(cout=256)
        #
        self.conv_high = conv(128, 128, stride=2, padding=1, kernel_size=3)
        #
        self.SE = SENet(in_chan=512, out_chan=512, stride=1) #60*80*c

    def forward(self, x):
        ##卷积+位置编码
        out = self.cnn_pre(x)
        out = out + self.pos_embed(out).cuda() ###???
        out_des = out 
        ##主干网络
        out_layer1 = self.layer1(out)#
        out_layer2 = self.layer2(out_layer1)
        out_layer3 = self.layer3(out_layer2)
        out_layer4 = self.layer4(out_layer3)
        ##1 c=512
        out_layer4_1 = self.SE(out_layer4)
        #2 新增
        out_layer_mul = self.multrans(out_des)
        # out_layer_mul = self.multrans(out_layer1)
        #3 新增
        out_layer_fuse = self.fusetrans(out_layer2, out_layer3, out_layer4)
        #通道注意力
        out = torch.cat([out_layer4_1, out_layer_mul, out_layer_fuse], dim=1)
        # out = out_layer4_1
        #回归
        out = self.last_conv(out)
        coord = self.coord_regress(out)
        uncer = self.uncer_regress(out)
        uncer = torch.sigmoid(uncer)
        
        return coord, uncer
if __name__ == "__main__":
    x = torch.randn(4,3,480,640)
    net = Multi_Scale_Trans()
    # net = Multi_Scale_Trans_v()
    coord, conf = net(x)
    print(coord.shape)
    print(conf.shape)














