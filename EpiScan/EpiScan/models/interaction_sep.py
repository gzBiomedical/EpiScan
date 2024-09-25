import numpy as np
import torch
import torch.functional as F
import torch.nn as nn

def corr2d(X, K):
    h, w = K.shape
    Y = torch.zeros(X.shape[0]-h+1, X.shape[1]-w+1)
    
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            
            Y[i,j] = (X[i:i+h, j:j+w]*K).sum()
    
    return Y

class LogisticActivation(nn.Module):

    def __init__(self, x0=0, k=1, train=False):
        super(LogisticActivation, self).__init__()
        self.x0 = x0
        self.k = nn.Parameter(torch.FloatTensor([float(k)]))
        self.k.requiresGrad = train

    def forward(self, x):
        o = torch.clamp(
            1 / (1 + torch.exp(-self.k * (x - self.x0))), min=0, max=1
        ).squeeze()
        return o

    def clip(self):
        self.k.data.clamp_(min=0)


class ModelInteraction(nn.Module):
    def __init__(
        self,
        embedding,
        embeddingAg,
        contact,
        use_cuda,
        do_w=True,
        do_sigmoid=True,
        do_pool=False,
        pool_size=9,
        theta_init=1,
        lambda_init=0,
        gamma_init=0,
    ):
        super(ModelInteraction, self).__init__()
        self.use_cuda = use_cuda
        self.do_w = do_w
        self.do_sigmoid = do_sigmoid
        if do_sigmoid:
            self.activation = LogisticActivation(x0=0.5, k=20)

        self.embedding = embedding
        self.embeddingAg = embeddingAg
        self.contact = contact
        # self.conv00 = nn.Conv1d(46, 1, 1)

        if self.do_w:
            self.theta = nn.Parameter(torch.FloatTensor([theta_init]))
            self.lambda_ = nn.Parameter(torch.FloatTensor([lambda_init]))

        self.do_pool = do_pool
        self.maxPool = nn.MaxPool2d(pool_size, padding=pool_size // 2)

        self.gamma = nn.Parameter(torch.FloatTensor([gamma_init]))

        self.clip()

    def clip(self):

        self.contact.clip()

        if self.do_w:
            self.theta.data.clamp_(min=0, max=1)
            self.lambda_.data.clamp_(min=0)

        self.gamma.data.clamp_(min=0)

    def embed(self, x):

        if self.embedding is None:
            return x
        else:
            return self.embedding(x)

    def embedAg(self, x):

        if self.embeddingAg is None:
            return x
        else:
            return self.embeddingAg(x)

    def cpred(self, z0, z1,catsite,cdrindex):


        e0 = torch.cat([z0[:,:,-43:], z0[:,:,-66:-63]], 2) 
        e0 = self.embedAg(e0)
        e1 = z1[:,:,-6165:]    
        e1 = self.embed(e1)


        B = self.contact.cmap(e0, e1,catsite,cdrindex)
        C = self.contact.predict(B)
        return C

    def map_predict(self, z0, z1,catsite,cdrindex): 

        C = self.cpred(z0, z1,catsite,cdrindex)

        if self.do_w:
            N, M = C.shape[2:]

            x1 = torch.from_numpy(
                -1
                * ((np.arange(N) + 1 - ((N + 1) / 2)) / (-1 * ((N + 1) / 2)))
                ** 2
            ).float()
            if self.use_cuda:
                x1 = x1.cuda()
            x1 = torch.exp(self.lambda_ * x1)

            x2 = torch.from_numpy(
                -1
                * ((np.arange(M) + 1 - ((M + 1) / 2)) / (-1 * ((M + 1) / 2)))
                ** 2
            ).float()
            if self.use_cuda:
                x2 = x2.cuda()
            x2 = torch.exp(self.lambda_ * x2)

            W = x1.unsqueeze(1) * x2
            W = (1 - self.theta) * W + self.theta

            yhat = C * W

        else:
            yhat = C

        if self.do_pool:
            yhat = self.maxPool(yhat)

        mu = torch.mean(yhat)
        sigma = torch.var(yhat)
        Q = torch.relu(yhat - mu - (self.gamma * sigma))
        phat = torch.sum(Q) / (torch.sum(torch.sign(Q)) + 1)
        if self.do_sigmoid:
            phat = self.activation(phat)
        return C, phat

    def predict(self, z0, z1):
        _, phat = self.map_predict(z0, z1)
        return phat

    def forward(self, z0, z1):

        return self.predict(z0, z1)
