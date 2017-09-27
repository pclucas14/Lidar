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
bbs = 30
batch_size = 16
use_cuda = True
noise = 2.
future = -1
C, H, W = 1, 128, 176
extra_dim = 0#  2
use_delta = False
mse_pretrain_epoch = 50

# gan parameters
lambdas = [5, 5, 10, 50, 1, .5, .5, 1]
seq_len, critic_iters, lambda_gp, lambda_mse, lambda_adv, lambda_extra, lambda_fake, lambda_real = lambdas
input_shape = (batch_size, seq_len, C, H, W)
version = 'kitti_lidar_delta_2' if use_delta else 'kitti_lidar_2'
load = ['kitti_lidar', '499']
target_index = slice(seq_len, seq_len+1) if future <= 0 else slice(seq_len, seq_len+future+1)

# NN creation
netG = RGEN(input_shape, gpu=gpu1,  extra_dim=extra_dim)
# netD = RDISC(input_shape, gpu=gpu1, extra_dim=extra_dim)

if use_cuda:
    netG = netG.cuda(gpu2)
    # netD = netD.cuda(gpu2)

if load is not None : 
    str_D = 'models/' + load[0] + 'netD_epoch_' + load[1] + '.pth'
    str_G = 'models/' + load[0] + 'netG_epoch_' + load[1] + '.pth'
    pdb.set_trace()
    netG.load_state_dict(torch.load(str_G))
    # netD.load_state_dict(torch.load(str_D))

optimizerD = optim.Adam(netD.parameters(), lr=2e-4, betas=(0.5, 0.9))
# optimizerG = optim.Adam(netG.parameters(), lr=2e-4, betas=(0.5, 0.9))

one = torch.FloatTensor([1])
mone = one * -1
if use_cuda:
    one = one.cuda(gpu2)
    mone = mone.cuda(gpu2)

for epoch in range(5000):
    if epoch % 200 == 0 : 
        future += 1
        datagen = load_kitti(bbs=bbs, batch_size=batch_size, seq_len=seq_len+future+1, skip=2)
    print 'loading data'
    bbatch, extra = next(datagen)
    stds = np.array([float(np.std(x)) for x in extra])
    data = iter_minibatches(bbatch, batch_size, extra=extra)
    print 'data loaded'
    real_d, fake_d, fake_g, mse_g, vgg_g, real_extra_d, fake_extra_d, fake_extra_g, i = [0] * 9
    critic_rounds, gen_rounds = 0, 0
    while i < bbs : 
        """(1) Update D network"""
        for p in netD.parameters():  
            p.requires_grad = True  
        
        j = 0
        bbs = 1
        critic_iter_epoch = 100 if epoch == 0 else critic_iters
        while j < critic_iters and i < bbs-1:
            _data, _extra = data.next()
            i += 1; j += 1
            
            _data = torch.from_numpy(_data)
            input, target = _data[:, :seq_len], _data[:, seq_len:]
            input = input.cuda(gpu1)
            target = target.cuda(gpu2)
            _extra = torch.cuda.FloatTensor(_extra)
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
            fake_out = fake_out.mean()
            fake_out.backward(one)

            # train with gradient penalty
            gradient_penalty = calc_gradient_penalty(netD_temp, input_v, target_v.data, 
                                                          target_v.data, gpu=gpu2, 
                                                          extra=extra_v,
                                                          fake_extra=old_extra,
                                                          batch_size=batch_size) * lambda_gp
            gradient_penalty.backward()
                
            
            # train with real
            _, real_out = netD(input_v, target_v, extra=extra_v) #, hid_=hid_)
            real_d += real_out.mean().data[0]
            real_out = real_out.mean()
            real_out.backward(mone)

            _, real_out = netD_temp(input_v, target_v, extra=extra_v) #, hid_=hid_)
            real_extra_d += real_out.mean().data[0]
            real_out = real_out.mean()
            real_out.backward(mone)

            # train with gradient penalty
            gradient_penalty = calc_gradient_penalty(netD, input_v, target_v.data, 
                                                      fake.data, gpu=gpu2, 
                                                      extra=extra_v,
                                                      batch_size=batch_size) * lambda_gp
            gradient_penalty.backward()

           

            optimizerD.step()
            optimizerD_temp.step()
            old_extra = extra_v
            critic_rounds += 1

        """ (2) Update G Network """
        for p in netD.parameters():
            p.requires_grad = False  # to avoid computation
        
        _data, _extra = data.next()
        i += 1
        
        _data = torch.from_numpy(_data)
        input, target = _data[:, :seq_len], _data[:, seq_len:]
        input = input.cuda(gpu1)
        target = target.cuda(gpu2)
        _extra = torch.cuda.FloatTensor(_extra)
        extra_v = Variable(_extra)
        
        netG.zero_grad()
        input_v, target_v = Variable(input), Variable(target)
        fake = netG(input_v, future=future, extra=extra_v)
        loss_mse = lambda_mse * ((fake - target_v + 1e-5) ** 2).mean()
        mse_g += loss_mse.data[0] * 1000
        
        if epoch < mse_pretrain_epoch : 
            loss_mse.backward()
        else : 
            _, fake_out = netD(input_v, fake, extra=extra_v)
            fake_g += fake_out.mean().data[0]
            loss_adv = lambda_adv * fake_out.mean() * -1
            loss_adv.backward(retain_graph=True)
            
            _, fake_out = netD_temp(input_v, fake, extra=extra_v)
            fake_extra_g += fake_out.mean().data[0]
            loss_adv = lambda_adv * fake_out.mean() * -1 
            loss_adv.backward()
         
        optimizerG.step()
        gen_rounds += 1
        
        
    iteration = epoch
    iters = bbs
    print iteration
    print lambdas
    print use_delta
    print("real out D %.5f" % (real_d / (critic_rounds)))
    print("fake out D %.5f" % (fake_d / (critic_rounds)))
    print("fake out G %.5f" % (fake_g / (gen_rounds)))
    print("real extra D %.5f" % (real_extra_d / (critic_rounds)))
    print("fake extra D %.5f" % (fake_extra_d / (critic_rounds)))
    print("fake extra G %.5f" % (fake_extra_g / (gen_rounds)))
    print("Wasserst D %.5f" % ((real_d - fake_d)))
    print("MSE g %.5f"      % (mse_g  / (iters)))
    print ""
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
    Image.fromarray(out).save('images/' + str(epoch) + '.png')
    if iteration % 50 == 49 : 
        torch.save(netG.state_dict(), '%snetG_epoch_%d.pth' % ('models/' + version, iteration))
        torch.save(netD.state_dict(), '%snetD_epoch_%d.pth' % ('models/' + version, iteration))
        print 'model saved'
       
