from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pdb
import torch.nn.functional as F


class STN3d(nn.Module):
    def __init__(self, num_points = 2500, dim=3):
        super(STN3d, self).__init__()
        self.num_points = num_points
        self.conv1 = torch.nn.Conv1d(dim, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        #self.mp1 = torch.nn.MaxPool1d(num_points)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.conv1(x)) 
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        #x = self.mp1(x)
        #print(x.size())
        x,_ = torch.max(x, 2)
        #print(x.size())
        x = x.view(-1, 1024)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class PointNetfeat(nn.Module):
    def __init__(self, num_points = 2500, global_feat = True, dim=3):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d(num_points = num_points, dim=dim)
        self.conv1 = torch.nn.Conv1d(dim, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)

        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(1024)

        self.num_points = num_points
        self.global_feat = global_feat

    def forward(self, x):
        batchsize = x.size()[0]
        trans = self.stn(x)
        # TODO : remove next 3 lines
        # x = x.transpose(2,1)
        # x = torch.bmm(x, trans)
        # x = x.transpose(2,1)
        x = F.relu(self.bn1(self.conv1(x)))
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x,_ = torch.max(x, 2)
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, self.num_points)
            return torch.cat([x, pointfeat], 1), trans

class PointNetCls(nn.Module):
    def __init__(self, num_points = 2500, k = 2):
        super(PointNetCls, self).__init__()
        self.num_points = num_points
        self.feat = PointNetfeat(num_points, global_feat=True)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.bn1 = torch.nn.BatchNorm1d(512)
        self.bn2 = torch.nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
    
    def forward(self, x, activation=True):
        x, trans = self.feat(x)
        global_feat = x
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        out = F.log_softmax(x) if activation else x
        return out, global_feat

class PointNetDenseCls(nn.Module):
    def __init__(self, num_points = 2500, k = 2):
        super(PointNetDenseCls, self).__init__()
        self.num_points = num_points
        self.k = k
        self.feat = PointNetfeat(num_points, global_feat=False)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)

    def forward(self, x):
        batchsize = x.size()[0]
        x, trans = self.feat(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        x = x.transpose(2,1).contiguous()
        x = F.log_softmax(x.view(-1,self.k))
        x = x.view(batchsize, self.num_points, self.k)
        return x, trans


class PointGen(nn.Module):
    def __init__(self, num_points = 2500):
        super(PointGen, self).__init__()
        self.num_points = num_points
        self.fc1 = nn.Linear(100, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, self.num_points * 3)

        self.th = nn.Tanh()
    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.th(self.fc4(x))
        x = x.view(batchsize, 3, self.num_points)
        return x


class PointGenC(nn.Module):
    def __init__(self, num_points = 2500):
        super(PointGenC, self).__init__()
        self.conv1 = nn.ConvTranspose1d(100, 1024, 2,2,0)
        self.conv2 = nn.ConvTranspose1d(1024, 512, 5,5,0)
        self.conv3 = nn.ConvTranspose1d(512, 256, 5,5,0)
        self.conv4 = nn.ConvTranspose1d(256, 128, 2,2,0)
        self.conv5 = nn.ConvTranspose1d(128, 64, 5,5,0)
        self.conv6 = nn.ConvTranspose1d(64, 3, 5,5,0)

        self.bn1 = torch.nn.BatchNorm1d(1024)
        self.bn2 = torch.nn.BatchNorm1d(512)
        self.bn3 = torch.nn.BatchNorm1d(256)
        self.bn4 = torch.nn.BatchNorm1d(128)
        self.bn5 = torch.nn.BatchNorm1d(64)
        self.th = nn.Tanh()
    def forward(self, x):

        batchsize = x.size()[0]
        x = x.view(-1, 100, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.conv6(x)

        x = self.th(x)
        return x


class PointGenPSG(nn.Module):
    def __init__(self, nz=100, num_points = 2048):
        super(PointGenPSG, self).__init__()
        self.num_points = num_points
        self.fc1 = nn.Linear(nz, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, (self.num_points-32*48) * 3)
        self.th = nn.Tanh()
        self.nz = nz
        
        self.conv1 = nn.ConvTranspose2d(nz,1024,(2,3))
        self.conv2 = nn.ConvTranspose2d(1024, 512, 4, 2, 1)
        self.conv3 = nn.ConvTranspose2d(512, 256, 4, 2, 1)
        self.conv4= nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.conv5= nn.ConvTranspose2d(128, 3, 4, 2, 1)
        
        self.bn1 = torch.nn.BatchNorm2d(1024)
        self.bn2 = torch.nn.BatchNorm2d(512)
        self.bn3 = torch.nn.BatchNorm2d(256)
        self.bn4 = torch.nn.BatchNorm2d(128)
        self.bn5 = torch.nn.BatchNorm2d(3)
        
        
        
    def forward(self, x):
        batchsize = x.size()[0]
        
        x1 = x
        x2 = x
        x1 = F.relu(self.fc1(x1))
        x1 = F.relu(self.fc2(x1))
        x1 = F.relu(self.fc3(x1))
        x1 = self.th(self.fc4(x1))
        x1 = x1.view(batchsize, 3, -1)#self.num_points / 4 * 1)
        x2 = x2.view(-1, self.nz, 1, 1)
        x2 = F.relu((self.conv1(x2)))
        x2 = F.relu((self.conv2(x2)))
        x2 = F.relu((self.conv3(x2)))
        x2 = F.relu((self.conv4(x2)))
        x2 = self.th((self.conv5(x2)))
        x2 = x2.view(-1, 3, 32 * 48)
        return torch.cat([x1, x2], 2)


class PointGenPSG2(nn.Module):
    def __init__(self, nz=100, num_points = 2048):
        super(PointGenPSG2, self).__init__()
        self.num_points = num_points
        self.fc1 = nn.Linear(nz, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, self.num_points * 3 / 2)

        self.fc11 = nn.Linear(nz, 256)
        self.fc21 = nn.Linear(256, 512)
        self.fc31 = nn.Linear(512, 1024)
        self.fc41 = nn.Linear(1024, self.num_points * 3 / 2)
        self.th = nn.Tanh()
        self.nz = nz
        
    
    def forward(self, x):
        batchsize = x.size()[0]
        
        x1 = x
        x2 = x
        x1 = F.relu(self.fc1(x1))
        x1 = F.relu(self.fc2(x1))
        x1 = F.relu(self.fc3(x1))
        x1 = self.th(self.fc4(x1))
        x1 = x1.view(batchsize, 3, -1)#self.num_points / 4 * 1)

        x2 = F.relu(self.fc11(x2))
        x2 = F.relu(self.fc21(x2))
        x2 = F.relu(self.fc31(x2))
        x2 = self.th(self.fc41(x2))
        x2 = x2.view(batchsize, 3, -1)#self.num_points / 4 * 1)

        return torch.cat([x1, x2], 2)

class ConvTBlock(nn.Module):
    def __init__(self, channels, nz=100, ks=(2,2), bn=False):
        super(ConvTBlock, self).__init__()
        layers = []
        
        # first layer
        layers.append(nn.ConvTranspose2d(nz, channels[0], ks))
        layers.append(nn.ReLU(True))
        
        # middle layers
        for i in range(1, len(channels)-1):
            layers.append(nn.ConvTranspose2d(channels[i-1], channels[i], 4, 2, 1))
            if bn : layers.append(nn.BatchNorm2d(channels[i]))
            layers.append(nn.ReLU(True))
        
        # last layer
        layers.append(nn.ConvTranspose2d(channels[-2], channels[-1], 4, 2, 1))
        layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        bs = x.size(0)
        x = self.main(x)
        return x.view(bs, 3, -1)


class PointGenPSG3(nn.Module):
    def __init__(self, nz=100, channels=[1024, 512, 256, 128, 3], num_blocks = 2, ks=(2,3), bn=False, use_linear=False):
        super(PointGenPSG3, self).__init__()
        self.blocks = nn.ModuleList([ConvTBlock(channels, nz=nz, bn=bn, ks=ks) for _ in range(num_blocks)])
        self.nz = nz
        self.num_blocks = num_blocks

        self.use_linear = use_linear
        if use_linear : 
            self.linear = nn.Sequential(
                nn.Linear(nz, 256),
                nn.ReLU(True), 
                nn.Linear(256, 512),
                nn.ReLU(True), 
                nn.Linear(512, 1024),
                nn.ReLU(True),
                nn.Linear(1024, 1024*3),
                nn.Tanh())

    def forward(self, x):
        x_conv = x.view(x.size(0), self.nz, 1, 1)
        xs = [self.blocks[i](x_conv) for i in range(self.num_blocks)]
        if self.use_linear : xs.append(self.linear(x).view(x.size(0), 3, -1))  
        return torch.cat(xs, 2)


class PointGen4096(nn.Module):
    def __init__(self, nz=100, channels=[1024, 512, 256, 128, 3], num_blocks=4, ks=(2,2), bn=False):
        super(PointGen4096, self).__init__()
        self.blocks = nn.ModuleList([ConvTBlock(channels, nz=nz, bn=bn, ks=ks) for _ in range(num_blocks)])
        self.nz = nz
        self.num_blocks = num_blocks

    def forward(self, x):
        x_conv = x.view(x.size(0), self.nz, 1, 1)
        xs = [self.blocks[i](x_conv) for i in range(self.num_blocks)]
        return torch.cat(xs, 2)


class PointGenPSG33(nn.Module):
    def __init__(self, nz=100, num_points = 2048):
        super(PointGenPSG33, self).__init__()
        self.num_points = num_points
        self.th = nn.Tanh()
        self.nz = nz
        
        self.conv1 = nn.ConvTranspose2d(nz,1024,(2,2))#3))
        self.conv2 = nn.ConvTranspose2d(1024, 512, 4, 2, 1)
        self.conv3 = nn.ConvTranspose2d(512, 256, 4, 2, 1)
        self.conv4= nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.conv5= nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.conv6= nn.ConvTranspose2d(64, 3, 4, 2, 1)
        
        self.bn1 = torch.nn.BatchNorm2d(1024)
        self.bn2 = torch.nn.BatchNorm2d(512)
        self.bn3 = torch.nn.BatchNorm2d(256)
        self.bn4 = torch.nn.BatchNorm2d(128)
        self.bn5 = torch.nn.BatchNorm2d(64)
        self.bn6 = torch.nn.BatchNorm2d(3)
        
        self.conv11 = nn.ConvTranspose2d(nz,1024,(2,2))#3))
        self.conv21 = nn.ConvTranspose2d(1024, 512, 4, 2, 1)
        self.conv31 = nn.ConvTranspose2d(512, 256, 4, 2, 1)
        self.conv41 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.conv51 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.conv61= nn.ConvTranspose2d(64, 3, 4, 2, 1)
        
        self.bn11 = torch.nn.BatchNorm2d(1024)
        self.bn21 = torch.nn.BatchNorm2d(512)
        self.bn31 = torch.nn.BatchNorm2d(256)
        self.bn41 = torch.nn.BatchNorm2d(128)
        self.bn51 = torch.nn.BatchNorm2d(64)
        self.bn61 = torch.nn.BatchNorm2d(3)
        
        
        
    def forward(self, x):
        batchsize = x.size()[0]
        
        x1 = x
        x2 = x
        x1 = x1.view(-1, self.nz, 1, 1)
        x1 = F.relu((self.conv11(x1)))
        x1 = F.relu((self.conv21(x1)))
        x1 = F.relu((self.conv31(x1)))
        x1 = F.relu((self.conv41(x1)))
        x1 = F.relu((self.conv51(x1)))
        x1 = self.th((self.conv61(x1)))
        x1 = x1.view(batchsize, 3, -1)
        
        x2 = x2.view(-1, self.nz, 1, 1)
        x2 = F.relu((self.conv1(x2)))
        x2 = F.relu((self.conv2(x2)))
        x2 = F.relu((self.conv3(x2)))
        x2 = F.relu((self.conv4(x2)))
        x2 = F.relu((self.conv5(x2)))
        x2 = self.th((self.conv6(x2)))
        x2 = x2.view(batchsize, 3, -1)
        return torch.cat([x1, x2], 2)


class PointGenPSG4(nn.Module):
    def __init__(self, nz=100, num_points = 2048):
        super(PointGenPSG4, self).__init__()
        self.num_points = num_points
        self.th = nn.Tanh()
        self.nz = nz
        
        self.conv1 = nn.ConvTranspose2d(nz,1024,(3,3))
        self.conv2 = nn.ConvTranspose2d(1024, 512, 4, 2, 1)
        self.conv3 = nn.ConvTranspose2d(512, 256, 4, 2, 1)
        self.conv4= nn.ConvTranspose2d(256, 128, 4, 2, 1)

        self.conv1_out = nn.ConvTranspose2d(1024, 3, 4 ,2, 1)
        self.conv2_out = nn.ConvTranspose2d(512 , 3, 4, 2, 1)
        self.conv3_out = nn.ConvTranspose2d(256 , 3, 4, 2, 1)
        self.conv4_out = nn.ConvTranspose2d(128 , 3, 4, 2, 1)
        
        self.bn1 = torch.nn.BatchNorm2d(1024)
        self.bn2 = torch.nn.BatchNorm2d(512)
        self.bn3 = torch.nn.BatchNorm2d(256)
        self.bn4 = torch.nn.BatchNorm2d(128)
        self.bn5 = torch.nn.BatchNorm2d(3)
        
        
    def forward(self, x):
        batch_size = x.size()[0]
        x = x.view(batch_size, self.nz, 1, 1)
        x1 = F.relu(self.conv1(x))
        x1_out = self.th(self.conv1_out(x1)).view(batch_size, 3, -1)

        x2 = F.relu(self.bn2(self.conv2(x1)))
        x2_out = self.th(self.conv2_out(x2)).view(batch_size, 3, -1)

        x3 = F.relu(self.bn3(self.conv3(x2)))
        x3_out = self.th(self.conv3_out(x3)).view(batch_size, 3, -1)

        x4 = F.relu(self.bn4(self.conv4(x3)))
        x4_out = self.th(self.conv4_out(x4)).view(batch_size, 3, -1)
        
        return torch.cat([x1_out, x2_out, x3_out, x4_out], 2)
