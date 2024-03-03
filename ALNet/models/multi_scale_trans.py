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

##
def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                  stride=stride, padding=padding),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True)
    )

#senet
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

#Similarity measure
#euclidean_distance
def compute_euclidean_similarity(batch):
    dists = euclidean_distance(batch)
    sim = 1 / (1 + dists) 
    return sim
def euclidean_distance(batch):
    batch_size, num_channels, num_points = batch.shape
    batch_reshaped = batch.reshape(batch_size, -1, num_channels) 
    dists = torch.sqrt(torch.sum((batch_reshaped[:, :, np.newaxis, :] - batch_reshaped[:, np.newaxis, :, :]) ** 2, axis=-1))
    return dists

##manhattan_distance
def compute_manhattan_similarity(batch):
    dists = manhattan_distance(batch)
    sim = 1 / (1 + dists) 
    return sim
def manhattan_distance(batch):
    batch_size, num_channels, num_points = batch.shape
    batch_reshaped = batch.reshape(batch_size, num_channels, -1) 
    distances = torch.abs(batch_reshaped[:, :, :, np.newaxis] - batch_reshaped[:, :, np.newaxis, :])
    manhattan_distances = torch.sum(distances, axis=1)  
    return manhattan_distances
#cosine_similarity
def compute_cosine_similarity(batch):
    batch_size, num_channels, num_points = batch.shape
    batch_reshaped = batch.reshape(batch_size, -1, num_channels)  #
    norms = torch.linalg.norm(batch_reshaped, axis=-1, keepdims=True)
    dot_products = torch.einsum('ijk,ihk->ijh', batch_reshaped, batch_reshaped)
    similarities = dot_products / (norms * torch.transpose(norms, 1, 2))
    return similarities

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
        B,C,H,W = feat.shape
        L = H*W
        window_size = 2*exp_size+1
        x1 = feat.reshape(B,C,L)
        similarity_matrix = torch.einsum('nci,ncj->nij', x1, x1)
        similarity_matrix = compute_euclidean_similarity(x1)
        #similarity_matrix = compute_manhattan_similarity(x1)
        #similarity_matrix = compute_cosine_similarity(x1)
        similarity_matrix = similarity_matrix.reshape(B,L,H,W)
        padding_size = (exp_size,exp_size,exp_size,exp_size) 
        pad_simi_matrix = F.pad(similarity_matrix, padding_size,'constant',1e20)

        b_index = torch.arange(B)[:,None,None].repeat(1,L,window_size**2)
        l_index = torch.arange(L)[None,:,None].repeat(B,1,window_size**2)
        I = torch.cat([self.make_i(i, exp_size) for i in range(exp_size,H+exp_size)],dim=0)
        i_index = I[None,:,None,:].repeat(B,1,W,1).reshape(B,L,window_size**2)
        J = torch.cat([self.make_j(j, exp_size) for j in range(exp_size,W+exp_size)],dim=0)
        j_index = J[None,None,:,:].repeat(B,H,1,1).reshape(B,L,window_size**2)

        win_value = pad_simi_matrix[b_index.long(),l_index.long(),i_index.long(),j_index.long()]

        top_values, top_indices = torch.topk(win_value,k=top_number,dim=2,largest=False)

        h_bias = top_indices // window_size-exp_size
        w_bias = top_indices %  window_size-exp_size

        center = torch.arange(L)[None, :, None].repeat(B, 1, top_number).to(h_bias.device)

        H_f = center // W + h_bias
        W_f = center % W + w_bias
        index = H_f * W + W_f

        x_stron = x1.permute(0,2,1)
        x_stron = x_stron[:,:,None,:].repeat(1,1,L,1)
        a_index = torch.arange(B)[:,None,None].repeat(1,L,top_number)
        b_index = torch.arange(L)[None,:,None].repeat(B,1,top_number)
        x_final = x_stron[a_index.long(),b_index.long(),index.long(),:] 
        return x_final
    ###
    def forward(self, x): ##x: n c h w 
        x1 = self.conv_down(x)
        # if self.se is not None:
        #     x1 = self.conv_se(x1)
        B,C,H,W = x1.shape
        L = H*W
        out_1 = self.get_feat(x1, self.exp_size, self.top_number)
        out_0 = x1.reshape(B,C,L).permute(0,2,1) 
        out_0 = out_0[:,:,None,:].repeat(1,1,1,1) 
        out = out_1 - out_0
        out = torch.mean(out, dim=2, keepdim=False) 
        out = out.permute(0,2,1).reshape(B,-1,H,W)
        out = torch.cat((x1, out), dim=1)
        out = self.conv(out)
        out = F.interpolate(out, size=[H*self.up_sample,W*self.up_sample], mode='bilinear', align_corners=False)
        return out

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
        )
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
        self.desimilar_block = Desimilar_block(exp_size=2,top_number=6,c=64,down_sample=8,up_sample=2)
        self.att_layer = att_layer(in_chan=(512+256+128),out_chan=512)
        self.conv_high = conv(128, 128, stride=2, padding=1, kernel_size=3)

    def forward(self, x):
        ##
        out = self.cnn_pre(x)
        out = out + self.pos_embed(out).cuda() ###???
        out_des = out 
        ##
        out_layer1 = self.layer1(out)#
        out_layer2 = self.layer2(out_layer1)
        out_layer3 = self.layer3(out_layer2)
        out_layer4 = self.layer4(out_layer3)
        ##
        out_layer4_1 = out_layer4
        #
        out_layer_des = self.desimilar_block(out_des)
        #
        out_layer2_2 = self.conv_high(out_layer2) 
        out = torch.cat([out_layer4_1, out_layer2_2, out_layer3], dim=1) 
        # 
        out = self.att_layer(out)
        out = torch.cat([out_layer_des, out_layer4_1, out], dim=1)
        #
        out = self.last_conv(out)
        coord = self.coord_regress(out)
        uncer = self.uncer_regress(out)
        uncer = torch.sigmoid(uncer)

        return coord, uncer
if __name__ == "__main__":
    x = torch.randn(4,3,480,640)
    net = Multi_Scale_Trans()
    coord, conf = net(x)
    print(coord.shape)
    print(conf.shape)














