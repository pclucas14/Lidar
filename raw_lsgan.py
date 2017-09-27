import os, sys
import cPickle
import time
import pdb
import hickle as hkl
import numpy as np
from utils import *
from models_2 import * 

# general parameters 
torch.manual_seed(1)
bbs = 100
batch_size = 16
nf = 64

# gan parameters
input_shape = (batch_size, 5000, 3)
load = ['raw_lidar_wgan', '399'] 
use_wgan = False
lambda_ortho = 0.001
critic_iters = 10 if use_wgan else 1
version = 'raw_lidar_wgan' if use_wgan else 'raw_lidar_lsgan'

netG = nnetG(1, ngf=nf).cuda() 
netD = PointNetfeat(num_points = 48*48).cuda() 

if load is not None : 
    str_D = 'models/' + load[0] + 'netD_epoch_' + load[1] + '.pth'
    str_G = 'models/' + load[0] + 'netG_epoch_' + load[1] + '.pth'
    netG.load_state_dict(torch.load(str_G))
    netD.load_state_dict(torch.load(str_D))

optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))
optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9))
datagen = load_kitti_lidar(bbs=bbs, batch_size=batch_size) #, train='train')

one = torch.FloatTensor([1]).cuda()
mone = one * -1

for epoch in range(5000):
    print 'loading data'
    bbatch = next(datagen)
    data = iter_minibatches(bbatch, batch_size)
    print 'data loaded'
    real_d, fake_d, fake_g, critic_rounds, gen_rounds, i = [0] * 6
    while i < bbs :
        """(1) Update D network"""
        for p in netD.parameters():  
            p.requires_grad = True
        for p in netG.parameters():
            p.requires_grad = False

        j = 0
        while j < critic_iters and i < bbs: 
            _data = data.next()
            _data = torch.from_numpy(_data)
            _data = _data.cuda()
            real_v = Variable(_data)
            i += 1; j += 1
            
            # train with real data 
            netD.zero_grad()
            real_out , trans = netD(real_v)
            real_d += real_out.mean().data[0]
            
            if use_wgan : 
                real_out = real_out.mean()
                real_out.backward(mone, retain_graph=True)
                ortho_reg = lambda_ortho * ortho_penalty(trans)
                ortho_reg.backward()
            else : 
                real_loss = torch.mean((real_out - 1)) ** 2
                real_loss = real_loss.mean()
                real_loss.backward()

            # train with fake data
            noise = torch.cuda.FloatTensor(batch_size, 100).normal_(0,1)
            noise_v = Variable(noise)
            fake = netG(noise_v)
            fake = fake.detach()
            fake_out, trans = netD(fake)
            fake_d += fake_out.mean().data[0]
            
            if use_wgan : 
                fake_out = fake_out.mean()
                fake_out.backward(one, retain_graph=True)
                ortho_reg = lambda_ortho * ortho_penalty(trans)
                ortho_reg.backward()
            else : 
                fake_loss = torch.mean((fake_out) ** 2)
                fake_loss.backward()

            if use_wgan : 
                grad_penalty = calc_gradient_penalty(netD, real_v.detach().data, 
                                              fake.detach().data, batch_size=batch_size) * 10
                grad_penalty.backward()
                grad_penalty = None

            optimizerD.step()
            critic_rounds += 1
        
        """ (2) Update G Network """
        for p in netD.parameters():
            p.requires_grad = False  
        for p in netG.parameters():
            p.requires_grad = True
        
        noise = torch.cuda.FloatTensor(batch_size, 100).normal_(0,1)
        noise_v = Variable(noise)
        fake = netG(noise_v)
        fake_out, _  = netD(fake)
        fake_g += fake_out.mean().data[0]

        if use_wgan : 
            fake_out = fake_out.mean()
            fake_out.backward(mone)
        else : 
            fake_loss = torch.mean((fake_out - 1) ** 2)
            fake_loss.backward()

        optimizerG.step()
        gen_rounds += 1
    
    print epoch
    print("real out D %.5f" % (real_d / (critic_rounds)))
    print("fake out D %.5f" % (fake_d / (critic_rounds)))
    print("fake out G %.5f" % (fake_g / (gen_rounds)))
    print ""

    fake_sample = fake.cpu().data.numpy()[0]
    lidar_to_img(fake_sample, epoch=epoch)
    real_sample = real_v.cpu().data.numpy()[0]
    lidar_to_img(real_sample, epoch=1000+epoch)
    if bbs < 5 : pdb.set_trace()
    
    if epoch % 50 == 49 : 
        torch.save(netG.state_dict(), '%snetG_epoch_%d.pth' % ('models/' + version, epoch))
        torch.save(netD.state_dict(), '%snetD_epoch_%d.pth' % ('models/' + version, epoch))
        print 'model saved'
