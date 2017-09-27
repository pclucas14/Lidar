import os, sys
import cPickle
import time
import pdb
import numpy as np
from utils import *
from models_2 import * 
# general parameters 
torch.manual_seed(1)
gpu1, gpu2, gpu_old = 0, 0, 0
bbs = 15# 50
batch_size = 16
use_cuda = True
noise = 2.
future = 3
C, H, W = 1, 128, 176
extra_dim = [1, 1]  # 2
use_delta = False
mse_pretrain_epoch = 100

# gan parameters
lambdas = [5, 3, 1, .5, 10, 25, .1, .5, .5, 4]
critic_iters, seq_len, lambda_real, lambda_extra, lambda_gp, lambda_mse, lambda_adv, lambda_extra, lambda_fake, lambda_real = lambdas
input_shape = (batch_size, seq_len, C, H, W)
version = 'lidar_lsgan_2'
load = ['lidar_lsgan', '2549']
target_index = (seq_len,seq_len+1) if future <= 0 else slice(seq_len, seq_len+future+1)

netG = RGEN(input_shape, gpu=gpu1, extra_dim=extra_dim)
netD = RDISC(input_shape, gpu=gpu1, extra_dim=extra_dim)

if use_cuda:
    netG = netG.cuda(gpu2)
    netD = netD.cuda(gpu2)

if load is not None : 
    str_D = 'models/' + load[0] + 'netD_epoch_' + load[1] + '.pth'
    str_G = 'models/' + load[0] + 'netG_epoch_' + load[1] + '.pth'
    netG.load_state_dict(torch.load(str_G))
    netD.load_state_dict(torch.load(str_D))

optimizerD = optim.Adam(netD.parameters(), lr=2e-4, betas=(0.5, 0.9))
optimizerG = optim.Adam(netG.parameters(), lr=2e-4, betas=(0.5, 0.9))
datagen = load_kitti(bbs=bbs, batch_size=batch_size, seq_len=seq_len+future+1, skip=2, train='train')

one = torch.FloatTensor([1])
mone = one * -1
if use_cuda:
    one = one.cuda(gpu2)
    mone = mone.cuda(gpu2)

if use_delta : future = future - 1
for epoch in range(5000):
    print 'loading data'
    bbatch, extra = next(datagen)
    stds = np.array([float(np.std(x)) for x in extra])
    data = iter_minibatches(bbatch, batch_size, extra=extra)
    print 'data loaded'
    real_d, fake_d, fake_g, mse_g , vgg_g, real_extra_d, fake_extra_d, fake_extra_g, i = [0] * 9 
    critic_rounds, gen_rounds = 0, 0
    # TODO : Remove this
    bbs = 1
    while i < bbs : 
        """(1) Update D network"""
        for p in netD.parameters():  
            p.requires_grad = True  
        
        j = 0
        while j < critic_iters and i < bbs-1:# and epoch > mse_pretrain_epoch: 
            _data, _extra = data.next()
            i += 1; j += 1
            
            _data = torch.from_numpy(_data)
            input, target = _data[:, :seq_len], _data[:, seq_len:]
            input = input.cuda(gpu1)
            target = target.cuda(gpu2)
            _extra = torch.cuda.FloatTensor(_extra)
            # TODO : remove this
            extra_v = Variable(_extra)

            
            input_v, target_v = Variable(input), Variable(target)
            fake = netG(input_v, future=future, extra=extra_v)
            fake = fake.cuda(gpu2)
            fake = fake.detach()
            netD.zero_grad()
            real_hidden = []

            # train with fake images
            hid_, fake_out = netD(input_v, fake, extra=extra_v)
            fake_d += fake_out.mean().data[0]
            fake_loss = torch.mean((fake_out) ** 2)
            fake_loss.backward()

            
            # train with real
            _, real_out = netD(input_v, target_v, extra=extra_v) #, hid_=hid_)
            real_d += real_out.mean().data[0]
            real_loss = lambda_real * torch.mean((real_out -1) ** 2)
            real_loss.backward()
            optimizerD.step()

            critic_rounds += 1
        
        """ (2) Update G Network """
        for p in netD.parameters():
            p.requires_grad = False  
        
        _data, _extra = data.next()
        i += 1
        
        _data = torch.from_numpy(_data)
        input, target = _data[:, :seq_len], _data[:, seq_len:]
        input = input.cuda(gpu1)
        target = target.cuda(gpu2)
        _extra = torch.cuda.FloatTensor(_extra)
        # extra_v = Variable(_extra)
        pdb.set_trace()

        extra_ = [Variable(torch.cuda.FloatTensor(input.size(0), _extra.size(1), 1).normal_(0,1))] * len(extra_dim)
        extra_v = [Variable(_extra[:, :, i].unsqueeze(2)) for i in range(2)]


        
        netG.zero_grad()
        input_v, target_v = Variable(input), Variable(target)
        fake = netG(input_v, future=future, extra=extra_v)
        loss_mse = lambda_mse * ((fake - target_v + 1e-5) ** 2).mean()
        mse_g += loss_mse.data[0] * 1000
        
        if epoch < mse_pretrain_epoch : 
            loss_mse.backward()
        else : 
            _, fake_out = netD(input_v, fake, extra=extra_v)
            loss_adv = lambda_adv * ((fake_out -1) ** 2).mean()
            fake_g += fake_out.mean().data[0]
            
            loss = loss_mse + loss_adv
            loss.backward()
         
        optimizerG.step()
        gen_rounds += 1
       

    iteration = epoch
    iters = bbs
    print iteration
    print lambdas
    print use_delta
    '''
    print("real out D %.5f" % (real_d / (critic_rounds)))
    print("fake out D %.5f" % (fake_d / (critic_rounds)))
    print("fake out G %.5f" % (fake_g / (gen_rounds)))
    print("real extra D %.5f" % (real_extra_d / (critic_rounds)))
    print("fake extra D %.5f" % (fake_extra_d / (critic_rounds)))
    print("fake extra G %.5f" % (fake_extra_g / (gen_rounds)))
    print("MSE g %.5f"      % (mse_g  / (iters)))
    print ""
    '''
    real_d, fake_d, fake_g, mse_g, vgg_g, real_extra_d, fake_extra_d, fake_extra_g = [0] * 8
   
    out = fake

    ''' for testing purposes only '''
    test_extra = extra_v
    for k in range(len(test_extra)):
        x = test_extra[k]
        x = 10 * x
    out_fake_noise = netG(input_v, future=future, extra=test_extra)
     
    input_v = input_v.squeeze(2)
    target_v = target_v.squeeze(2)
    out = out.squeeze(2)
    out_fake_noise = out_fake_noise.squeeze(2)
    try_something = torch.cat([target_v, out, out_fake_noise], dim=1)
    out = [try_something[i] for i in range(try_something.size(0))]
    out = torch.cat(out, dim=1)
    out = [out[i] for i in range(out.size(0))]
    out = torch.cat(out, dim=1)
    out = out.cpu().data.numpy() * 255.
    out = out.astype('uint8')
    img = Image.fromarray(out)
    img.save('images/' + str(epoch) + '.png')
    pdb.set_trace()
    if iteration % 50 == 49 : 
        torch.save(netG.state_dict(), '%snetG_epoch_%d.pth' % ('models/' + version, iteration))
        torch.save(netD.state_dict(), '%snetD_epoch_%d.pth' % ('models/' + version, iteration))
        print 'model saved'
