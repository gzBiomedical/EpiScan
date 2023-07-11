
import torch
import torch.nn as nn
# import torch.functional as F
import torch.nn.functional as F

import torch.autograd as autograd

import sys,os
sys.path.append(os.getcwd())

from EpiScan.selfLoss.attBlock import se_block,eca_block,Rose_block,rotate_matTorch
from EpiScan.selfLoss.crossAtt import CrossAttention



class FullyConnected(nn.Module):

    def __init__(self, embed_dim, hidden_dim, activation=nn.ReLU()):
        super(FullyConnected, self).__init__()

        self.D = embed_dim
        self.H = hidden_dim
        self.sigLayer = nn.Sigmoid()
        # self.corr2dLayer = corr2d()
        self.H2 = int(hidden_dim/2)                   #not exist
        self.convz0z1 = nn.Conv2d(1, 2*self.D, 1)
        self.convz0 = nn.Conv1d(1, self.D, 1)
        self.convz1 = nn.Conv1d(1, self.D, 1)
        self.conv000 = nn.Conv1d(1, self.D, 1)
        self.conv111 = nn.Conv1d(1, self.D, 1)
        self.conv00 = nn.Conv1d(self.D, 1, 1)
        self.conv11 = nn.Conv1d(self.D, 1, 1)
        self.conv0 = nn.Conv2d(1, 1, 3,padding = 1)
        self.conv = nn.Conv2d(2 * self.D, self.H, 1)
        self.conv2 = nn.Conv2d(self.H, self.H2, 1)    #not exist
        self.batchnorm = nn.BatchNorm2d(self.H)      #self.H
        self.batchnorm2 = nn.BatchNorm2d(self.H2)      #self.H
        self.activation = activation
        self.se_block = eca_block(self.D)
        self.se_block2 = eca_block(self.D)
        self.se_block3 = eca_block(self.D)
        self.se_block4 = eca_block(self.D)
        self.se_block5 = eca_block(self.D)
        self.se_block6 = eca_block(self.D)
        self.se_block7 = eca_block(self.D)
        self.se_block8 = eca_block(self.D)
        self.se_block9 = eca_block(self.D)
        self.ERose_block = Rose_block(self.D,3)    ###for seq
        self.BRose_block = Rose_block(self.D,3)    ####now for Rotate
        self.Rose_block1 = Rose_block(self.D,3)
        self.Rose_block2 = Rose_block(self.D,3)
        self.ARose_block1 = Rose_block(self.D,1)
        self.ARose_block2 = Rose_block(self.D,1)
        self.cross_attention = CrossAttention(46, 256)

    def forward(self, z0, z1, catsite, cdrindex):

        # # z0 is (b,N,d), z1 is (b,M,d)
        z0 = z0.transpose(1, 2)
        z1 = z1.transpose(1, 2)
        # # z0 is (b,d,N), z1 is (b,d,M)


        #####20221117 Rotation code [start]
        bias_z1 = self.BRose_block(z1)
        meanbias_z1 =torch.mean(bias_z1,2).unsqueeze(2)    #(batch,3,1)
        axis_z11 = self.Rose_block1(z1)
        meanaxis_z11 =torch.mean(axis_z11,2).unsqueeze(2)  #(batch,3,1)
        ang_z11 = self.ARose_block1(z1)
        # meanang_z11 = torch.mean(ang_z11,(1,2))     #(batch)
        meanang_z11 =self.sigLayer(torch.mean(ang_z11,(1,2))) 
        coorz1 = z1[:,-3:,:]+meanbias_z1             #(batch,3,Len)
        Rocoorz1 = rotate_matTorch(coorz1,meanaxis_z11,meanang_z11)
        z1end = torch.cat([z1[:,:-3,:], Rocoorz1], 1) 



        if catsite <0 :
            z1L = z1end[:,:,:-1*catsite]   ####
            z1H = z1end[:,:,-1*catsite:]
        else:
            z1H = z1end[:,:,:catsite]    ## -5-5-5-5
            z1L = z1end[:,:,catsite:]
        #####20221117 Rotation code [end]  
    
               

        if catsite <0 :
            z1L = z1[:,:,:-1*catsite]   ####
            z1H = z1[:,:,-1*catsite:]
        else:
            z1H = z1[:,:,:catsite]    ## -5-5-5-5
            z1L = z1[:,:,catsite:]
                 
        cdrHind = [i for i in cdrindex if i <= len(z1H[0,0,:])]
        cdrLind = [i-len(z1H[0,0,:]) for i in cdrindex if i > len(z1H[0,0,:])]
        indAll = [i for i in range(len(z1[0,0,:])-1)]
        indnotcdr = [i for i in indAll if i not in cdrindex]
        notcdrHind = [i for i in indnotcdr if i < len(z1H[0,0,:])]
        notcdrLind = [i-len(z1H[0,0,:]) for i in indnotcdr if i >= len(z1H[0,0,:])]


        z0 = self.se_block(z0)     #### attBlock in here for try
        z1H = self.se_block2(z1H)
        z1L = self.se_block3(z1L)     #### attBlock in here for try

    
        if 1+1 == 2:
            ##------------------ CDR cat----------------------------------##
            # cdrHind = torch.tensor([[cdrHind]]).cuda().repeat(1,46,1)
            # cdrLind = torch.tensor([[cdrLind]]).cuda().repeat(1,46,1)
            # notcdrHind = torch.tensor([[notcdrHind]]).cuda().repeat(1,46,1)
            # notcdrLind = torch.tensor([[notcdrLind]]).cuda().repeat(1,46,1)

            cdrHind = torch.tensor([[cdrHind]]).cuda()
            cdrLind = torch.tensor([[cdrLind]]).cuda()
            notcdrHind = torch.tensor([[notcdrHind]]).cuda()
            notcdrLind = torch.tensor([[notcdrLind]]).cuda()
            # print(cdrHind)
            # print(cdrLind)
            # Z1Hcdr = z1H[:,:,cdrHind]
            # Z1Hnotcdr = z1H[:,:,notcdrHind]
            # Z1Lcdr = z1L[:,:,cdrLind]
            # Z1Lnotcdr = z1L[:,:,notcdrLind]
            Z1Hcdr = torch.gather(z1H, 2, cdrHind)
            Z1Hnotcdr = torch.gather(z1H, 2, notcdrHind)
            Z1Lcdr = torch.gather(z1L, 2, cdrLind)
            Z1Lnotcdr = torch.gather(z1L, 2, notcdrLind)
            # print('rrrrrrrrccdd')
            # print(cdrHind)
            # print(Z1Hcdr.shape)

            # print('Z1Hcdr',Z1Hcdr)
            # print('Z1Hnotcdr',Z1Hnotcdr)

            z_Hdifcdr = torch.abs(Z1Hcdr.unsqueeze(3) - Z1Hnotcdr.unsqueeze(2))
            # z_Hmulcdr = Z1Hcdr.unsqueeze(3) * Z1Hnotcdr.unsqueeze(2)
            # print('z_Hmulcdr',z_Hmulcdr[:,:,:4,:])
            # z_Hcatcdr = torch.cat([z_Hdifcdr, z_Hmulcdr], 1)
            z_Hmulcdr = self.se_block4(z_Hdifcdr)
            z_Hcdr_mean = torch.mean(z_Hmulcdr,3)
            # z_Hcdr_mean = self.se_block4(z_Hcdr_mean)
            # z_Hcdr_mean = torch.squeeze(z_Hcdr_mean) 
            z1H = z_Hcdr_mean
            # print(z1H.shape)

            z_Ldifcdr = torch.abs(Z1Lcdr.unsqueeze(3) - Z1Lnotcdr.unsqueeze(2))
            # z_Lmulcdr = Z1Lcdr.unsqueeze(3) * Z1Lnotcdr.unsqueeze(2)
            # z_Lcatcdr = torch.cat([z_Ldifcdr, z_Lmulcdr], 1)
            z_Lmulcdr = self.se_block5(z_Ldifcdr)
            z_Lcdr_mean = torch.mean(z_Lmulcdr,3)
            # z_Lcdr_mean = self.se_block5(z_Lcdr_mean)
            # z_Lcdr_mean = torch.squeeze(z_Lcdr_mean) 
            z1L = z_Lcdr_mean

            
            ##------------------ CDR cat----------------------------------##

        

        z_Hdif = torch.abs(z0.unsqueeze(3) - z1H.unsqueeze(2))
        z_Hmul = z0.unsqueeze(3) * z1H.unsqueeze(2)
        # z_Hdif = self.se_block6(z_Hdif)     #### attBlock in here for try
        # z_Hmul = self.se_block7(z_Hmul)     #### attBlock in here for try
        # print(z_dif.shape)
        # print(z_mul.shape)
        z_Hcat = torch.cat([z_Hdif, z_Hmul], 1)   
        # z_Hcat = self.se_block6(z_Hcat)  
        # print(z_cat.shape)
        z0_mean = torch.mean(z_Hcat,3)
        halfLen = int(z0_mean.shape[1]/2)
        z0_Ldif = z0_mean[:,:halfLen,:]
        z0_Lmul = z0_mean[:,halfLen:,:]
        z_Ldif = torch.abs(z0_Ldif.unsqueeze(3) - z1L.unsqueeze(2))
        z_Lmul = torch.abs(z0_Lmul.unsqueeze(3) - z1L.unsqueeze(2))
        # z_Ldif = self.se_block8(z_Ldif)     #### attBlock in here for try
        # z_Lmul = self.se_block9(z_Lmul)
        z_Lcat = torch.cat([z_Ldif, z_Lmul], 1) 
        # z_Lcat = self.se_block7(z_Lcat)
        z_cat = torch.cat([z_Hcat, z_Lcat], 3)
        # gd= 2*gd


        # print(z_cat.shape)    
        # pj = pj*2      
        c = self.conv(z_cat)
        c = self.activation(c)
        c = self.batchnorm(c)
        # c = self.se_block2(c)          #### attBlock in here for try
        # print(c.shape)
        c = self.conv2(c)
        # print('llllll',c.shape)
        c = self.activation(c)
        c = self.batchnorm2(c)

        return c


class ContactCNN(nn.Module):

    def __init__(
        self, embed_dim, hidden_dim, width, activation=nn.Sigmoid()  ##hidden_dim=50, width=7,
    ):
        super(ContactCNN, self).__init__()

        self.hidden = FullyConnected(embed_dim, hidden_dim)
        hidden_dim2 = hidden_dim/2
        self.conv = nn.Conv2d(int(hidden_dim2), 1, width, padding=width // 2)    #hidden_dim not/2
        self.batchnorm = nn.BatchNorm2d(1)
        self.activation = activation
        self.clip()

    def clip(self):
        w = self.conv.weight
        self.conv.weight.data[:] = 0.5 * (w + w.transpose(2, 3))

    def forward(self, z0, z1):
        C = self.cmap(z0, z1)
        return self.predict(C)

    def cmap(self, z0, z1, catsite,cdrindex):
        C = self.hidden(z0, z1, catsite,cdrindex)
        return C

    def predict(self, C):

        s = self.conv(C)
        # print(s.shape)
        s = self.batchnorm(s)
        s = self.activation(s)
        # print(s.shape)
        return s




