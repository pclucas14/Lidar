import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.autograd as autograd
import torch.optim as optim
import torchvision
import numpy as np
import pdb
from utils import *



class nnetG(nn.Module):
    def __init__(self, ngpu, nz=100, ngf=64, nc=3, base=4, ff=(3,16)):
        super(nnetG, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(     nz, ngf * 8, ff, 1, 0, bias=False), # 3 was a 4
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),
                # state size. (ngf*8) x 4 x 4
                nn.ConvTranspose2d(ngf * 8, ngf * 4, (3,4), stride=2, padding=(0,1), bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
                # state size. (ngf*4) x 8 x 8
                nn.ConvTranspose2d(ngf * 4, ngf * 2, (3,4), 2, padding=(0,1), bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
                # state size. (ngf*2) x 16 x 16
                nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),
                # state size. (ngf) x 32 x 32
                nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
                #nn.Tanh()
                nn.Sigmoid()
                # state size. (nc) x 64 x 64
                )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            input = input.unsqueeze(2).unsqueeze(3)
            output = self.main(input)
        return output



class nnetD(nn.Module):
    def __init__(self, ngpu, ndf=64, nc=3, nz=1, lf=(3,16)):
        super(nnetD, self).__init__()
        self.encoder = True if nz > 1 else False
        self.ngpu = ngpu
        self.main = nn.Sequential(
                # input is (nc) x 64 x 64
                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf) x 32 x 32
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*2) x 16 x 16
                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*4) x 8 x 8
                nn.Conv2d(ndf * 4, ndf * 8, (3,4), 2, (0,1), bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*8) x 4 x 4
                )
        self.main_ = nn.Sequential(
                nn.Conv2d(ndf * 8, nz, lf, 1, 0, bias=False))

    def forward(self, input, return_hidden=False):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            if input.size(-1) == 3: input = input.transpose(1, 3)
            output_tmp = self.main(input)
            output = self.main_(output_tmp)
        if return_hidden : 
            return output, output_tmp
        return output if self.encoder else output.view(-1, 1).squeeze(1) 


class transNet(nn.Module):
    def __init__(self, z_dim=100, u_dim=4, h_dim=100, bs=32, save_mem=True, dropout=True):
        super(transNet, self).__init__()
        self.z_dim = z_dim
        self.u_dim = u_dim
        self.save_mem = save_mem

        layers = []
        layers.append(nn.Linear(z_dim, h_dim))
        layers.append(nn.ReLU())
        if dropout : 
            layers.append(nn.Dropout())
        
        layers.append(nn.Linear(h_dim, h_dim)) 
        layers.append(nn.ReLU()) 
        if dropout : 
            layers.append(nn.Dropout())
        
        layers.append(nn.Linear(h_dim, z_dim))
        layers.append(nn.ReLU())

        self.main = nn.Sequential(*layers) # bs x z_dim --> bs x z_dim
        if save_mem : 
            self.h_to_A = nn.Linear(z_dim, z_dim * 2)
            self.I = Variable(torch.eye(z_dim).view(1, z_dim, z_dim).repeat(bs, 1, 1),
                          requires_grad=False).cuda()
        else : 
            self.h_to_A = nn.Linear(z_dim, z_dim ** 2)

        self.h_to_B = nn.Linear(z_dim, z_dim * u_dim)
        self.h_to_o = nn.Linear(z_dim, z_dim)

    def forward(self, z, u):
        h = self.main(z)
        if self.save_mem : 
            vr = self.h_to_A(h).view(-1, self.z_dim * 2)
            v, r = vr.split(self.z_dim, dim=-1)
            A = (self.I + torch.bmm(v.unsqueeze(-1), r.unsqueeze(1)))
        else : 
            A = self.h_to_A(h).view(-1, self.z_dim, self.z_dim)

        B = self.h_to_B(h).view(-1, self.z_dim, self.u_dim)
        o = self.h_to_o(h).view(-1, self.z_dim, 1)
        z = z.unsqueeze(-1)
        u = u.unsqueeze(-1)
        Az = torch.bmm(A, z)
        Bu = torch.bmm(B, u)
        out = Az + Bu + o
        return out, v, r if self.save_mem else out


if __name__ == '__main__' : 
    net = transNet(bs=64).cuda()
    data = Variable(torch.cuda.FloatTensor(64, 100))
    actions = Variable(torch.cuda.FloatTensor(64, 4))
    z_tp1, v, r = net(data, actions)

        


