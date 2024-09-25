#-*- encoding:utf8 -*-

import os
import time
import sys

import torch as t
from torch import nn
from torch.autograd import Variable
import math


#from basic_module import BasicModule
from EpiScan.models.BasicModule import BasicModule
from EpiScan.selfLoss.attBlock import se_block,eca_block


sys.path.append("../")
# from utils.config import DefaultConfig
# configs = DefaultConfig()


class weight_cal(nn.Module):
    def __init__(self,in_features):
        super(weight_cal,self).__init__()
        self.in_features = in_features
        self.weight = nn.Parameter(t.Tensor(self.in_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1./math.sqrt(self.weight.size(0))
        # self.weight.data.uniform_(-stdv,stdv)
        self.weight.data.uniform_(0,stdv)

    def forward(self,input1,input2):
        x = input1*self.weight[0]+input2*self.weight[1]
        # x = input1*self.weight+input2*(1-self.weight)
        # print('-----------------wwwwwwwwwwwwwwwwwwwww------------')
        # print(self.weight[0])
        # print(self.weight[1])
        # x = x.sum(dim=1,keepdim=True)
        return x



class ConvsLayer(BasicModule):
    def __init__(self,):

        super(ConvsLayer,self).__init__()
        
        self.kernels = [13,15,17]   #configs.kernels
        hidden_channels = 4  #configs.max_sequence_length
        in_channel = 1
        features_L =  50  #configs.max_sequence_length
        seq_dim = 46
        # dssp_dim = configs.dssp_dim
        # pssm_dim = configs.pssm_dim
        W_size = seq_dim

        padding1 = (self.kernels[0]-1)//2
        padding2 = (self.kernels[1]-1)//2
        padding3 = (self.kernels[2]-1)//2
        self.conv1 = nn.Sequential()
        self.conv1.add_module("conv1",
            nn.Conv2d(in_channel, hidden_channels,
            padding=(padding1,0),
            kernel_size=(self.kernels[0],W_size)))
        self.conv1.add_module("ReLU",nn.PReLU())
        # self.conv1.add_module("pooling1",nn.MaxPool2d(kernel_size=(features_L,1),stride=1))
        
        self.conv2 = nn.Sequential()
        self.conv2.add_module("conv2",
            nn.Conv2d(in_channel, hidden_channels,
            padding=(padding2,0),
            kernel_size=(self.kernels[1],W_size)))
        self.conv2.add_module("ReLU",nn.ReLU())
        # self.conv2.add_module("pooling2",nn.MaxPool2d(kernel_size=(features_L,1),stride=1))
        
        self.conv3 = nn.Sequential()
        self.conv3.add_module("conv3",
            nn.Conv2d(in_channel, hidden_channels,
            padding=(padding3,0),
            kernel_size=(self.kernels[2],W_size)))
        self.conv3.add_module("ReLU",nn.ReLU())
        # self.conv3.add_module("pooling3",nn.MaxPool2d(kernel_size=(features_L,1),stride=1))

    
    def forward(self,x):

        features1 = self.conv1(x)
        features2 = self.conv2(x)
        features3 = self.conv3(x)
        features = t.cat((features1,features2,features3),1)
        features = features.unsqueeze(0)
        shapes = features.data.shape
        features = features.view(shapes[0],shapes[1]*shapes[2])   #features = features.view(shapes[0],shapes[1]*shapes[2]*shapes[3])
        
        return features





class DeepPPI(BasicModule):
    def __init__(self,class_nums,window_size,ratio=None):
        super(DeepPPI,self).__init__()
        # global configs
        # configs.kernels = [13, 15, 17]
        self.dropout =  0.2   #configs.dropout =

        seq_dim = 46*50  # configs.seq_dim configs.max_sequence_length
        
        
        self.seq_layers = nn.Sequential()
        self.seq_layers.add_module("seq_embedding_layer",
        nn.Linear(seq_dim,seq_dim))
        self.seq_layers.add_module("seq_embedding_ReLU",
        nn.ReLU())


        # seq_dim = 106 #configs.seq_dim
        # dssp_dim = configs.dssp_dim
        # pssm_dim = configs.pssm_dim
        local_dim = (window_size*2+1)*(seq_dim)
        if ratio:
            cnn_chanel = (local_dim*int(ratio[0]))//(int(ratio[1])*3)
        else:
            cnn_chanel = 4
        input_dim = cnn_chanel*150

        self.multi_CNN = nn.Sequential()
        self.multi_CNN.add_module("layer_convs",
                               ConvsLayer())

        
        self.DNN1 = nn.Sequential()
        self.DNN1.add_module("DNN_layer1",
                            nn.Linear(input_dim,512))
        self.DNN1.add_module("ReLU1",
                            nn.ReLU())
        # self.DNN1.add_module("BN1",
        #             nn.BatchNorm2d(512))
        #self.dropout_layer = nn.Dropout(self.dropout)
        self.DNN2 = nn.Sequential()
        self.DNN2.add_module("DNN_layer2",
                            nn.Linear(512,256))
        self.DNN2.add_module("ReLU2",
                            nn.ReLU())
        # self.DNN2.add_module("BN2",
        #             nn.BatchNorm2d(256))
                            


        self.outLayer = nn.Sequential(
            nn.Linear(256, class_nums),
            nn.Sigmoid())

        self.wcal = weight_cal((1,2))
        # self.wcal = weight_cal(1)

        self.sigLayer = nn.Sigmoid()

        self.se_block = eca_block(50)

    def forward(self,p0Conseq,prob_map,flag=0):
        if flag == 1:
            phatSeqall = t.tensor([[]]).cuda()
            for jj in range(p0Conseq.shape[1]//50):
                seq = p0Conseq[:,(jj*50):((jj+1)*50),-46:]
                # print('---------1111111111111111111111111111------------')
                shapes = seq.data.shape
                features = seq.contiguous().view(shapes[0],shapes[1]*shapes[2])
                features = self.seq_layers(features)
                features = features.contiguous().view(shapes[0],shapes[1],shapes[2])   ####embedding
                # features = t.cat((features,dssp,pssm),3)
                features = self.multi_CNN(features)
                # features = t.cat((features, local_features), 1)
                features = self.DNN1(features)
                #features =self.dropout_layer(features)
                features = self.DNN2(features)
                features = self.outLayer(features)
                # print(features)
                features = features.unsqueeze(0)
                features =  self.se_block(features)
                features = features.squeeze(0)
                phatSeqall = t.cat((phatSeqall, features), 1)
            phatSeqcat = phatSeqall[0][:(prob_map.shape[0])]   ####keysss
            # print('---------1111111111111111111111111111------------')
            # print(phatSeqcat)
            # print('---------2222222222222222222222222222------------')
            # print(prob_map)
            phatnew = self.wcal(phatSeqcat,prob_map)
            # phatnew = phatSeqall[0] #phatSeqcat
            phatnew = self.sigLayer(phatnew)
        phatnew = self.sigLayer(prob_map)
        # phatnew = prob_map

        if flag == 1:
            return phatnew,phatSeqcat #phatSeqcat
        else:
            return phatnew,phatnew#phatSeqcat


import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv1DNet(nn.Module):
    def __init__(self, input_shape):
        super(Conv1DNet, self).__init__()
        self.conv1 = nn.Conv1d(input_shape, 32, 3)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, 3)
        self.pool2 = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(64, 128)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x

class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet3D, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2, 2)
        )

        self.middle = nn.Sequential(
            nn.Conv3d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2, 2)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 2, stride=2),
            nn.ReLU(),
            nn.Conv3d(32, out_channels, 3, padding=1)
        )

    def forward(self, x):
        x1 = self.encoder(x)
        x2 = self.middle(x1)
        x3 = self.decoder(x2)
        return x3

class FusionModel(nn.Module):
    def __init__(self, input_shape_1d, input_shape_3d, out_channels_3d):
        super(FusionModel, self).__init__()
        self.conv1d_net = Conv1DNet(input_shape_1d)
        self.unet3d = UNet3D(input_shape_3d, out_channels_3d)

    def forward(self, x1, x2):
        x1_out = self.conv1d_net(x1)
        x2_out = self.unet3d(x2)

        # Fusion
        x1_out = x1_out.view(x1_out.size(0), -1, 1, 1, 1)
        fusion_out = x1_out * x2_out

        return fusion_out
