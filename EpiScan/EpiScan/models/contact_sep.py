
import torch
import torch.nn as nn
# import torch.functional as F
import torch.nn.functional as F

import torch.autograd as autograd

import sys,os
sys.path.append(os.getcwd())

from EpiScan.selfLoss.attBlock import se_block,eca_block,Rose_block,rotate_matTorch
from EpiScan.selfLoss.crossAtt import CrossAttention




def corr2d(X, K):
    h, w = K.shape
    Y = torch.zeros(X.shape[0]-h+1, X.shape[1]-w+1)
    
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            
            Y[i,j] = (X[i:i+h, j:j+w]*K).sum()
    
    return Y




class FullyConnected(nn.Module):


    def __init__(self, embed_dim, hidden_dim, activation=nn.ReLU()):
        super(FullyConnected, self).__init__()

        self.D = embed_dim
        self.H = hidden_dim
        self.sigLayer = nn.Sigmoid()
        # self.corr2dLayer = corr2d()
        self.H2 = int(hidden_dim/2)                   
        self.conv = nn.Conv2d(2 * self.D, self.H, 1)
        self.conv2 = nn.Conv2d(self.H, self.H2, 1)   
        self.batchnorm = nn.BatchNorm2d(self.H)     
        self.batchnorm2 = nn.BatchNorm2d(self.H2)     
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

    def forward(self, z0, z1, catsite, cdrindex):

        z0 = z0.transpose(1, 2)
        z1 = z1.transpose(1, 2)
        # z0 is (b,d,N), z1 is (b,d,M)

        

        if catsite <0 :
            z1L = z1[:,:,:-1*catsite] 
            z1H = z1[:,:,-1*catsite:]
        else:
            z1H = z1[:,:,:catsite]  
            z1L = z1[:,:,catsite:]
                 
        cdrHind = [i for i in cdrindex if i <= len(z1H[0,0,:])]
        cdrLind = [i-len(z1H[0,0,:]) for i in cdrindex if i > len(z1H[0,0,:])]
        indAll = [i for i in range(len(z1[0,0,:])-1)]
        indnotcdr = [i for i in indAll if i not in cdrindex]
        notcdrHind = [i for i in indnotcdr if i < len(z1H[0,0,:])]
        notcdrLind = [i-len(z1H[0,0,:]) for i in indnotcdr if i >= len(z1H[0,0,:])]




        z0 = self.se_block(z0)    
        z1H = self.se_block2(z1H)
        z1L = self.se_block3(z1L)   

    
        if 1+1 == 2:
            ##------------------ CDR cat----------------------------------##
            cdrHind = torch.tensor([[cdrHind]]).cuda().repeat(1,46,1)
            cdrLind = torch.tensor([[cdrLind]]).cuda().repeat(1,46,1)
            notcdrHind = torch.tensor([[notcdrHind]]).cuda().repeat(1,46,1)
            notcdrLind = torch.tensor([[notcdrLind]]).cuda().repeat(1,46,1)
            Z1Hcdr = torch.gather(z1H, 2, cdrHind)
            Z1Hnotcdr = torch.gather(z1H, 2, notcdrHind)
            Z1Lcdr = torch.gather(z1L, 2, cdrLind)
            Z1Lnotcdr = torch.gather(z1L, 2, notcdrLind)


            z_Hdifcdr = torch.abs(Z1Hcdr.unsqueeze(3) - Z1Hnotcdr.unsqueeze(2))
            z_Hmulcdr = self.se_block4(z_Hdifcdr)
            z_Hcdr_mean = torch.mean(z_Hmulcdr,3)
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

        z_Hdif = torch.abs(z0.unsqueeze(3) - z1H.unsqueeze(2))
        z_Hmul = z0.unsqueeze(3) * z1H.unsqueeze(2)
        z_Hcat = torch.cat([z_Hdif, z_Hmul], 1)   

        z0_mean = torch.mean(z_Hcat,3)
        halfLen = int(z0_mean.shape[1]/2)
        z0_Ldif = z0_mean[:,:halfLen,:]
        z0_Lmul = z0_mean[:,halfLen:,:]
        z_Ldif = torch.abs(z0_Ldif.unsqueeze(3) - z1L.unsqueeze(2))
        z_Lmul = torch.abs(z0_Lmul.unsqueeze(3) - z1L.unsqueeze(2))
        z_Lcat = torch.cat([z_Ldif, z_Lmul], 1) 
        # z_Lcat = self.se_block7(z_Lcat)
        z_cat = torch.cat([z_Hcat, z_Lcat], 3) 
     
        c = self.conv(z_cat)
        c = self.activation(c)
        c = self.batchnorm(c)
        # c = self.se_block2(c)    
        # print(c.shape)
        c = self.conv2(c)
        # print(c.shape)
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




