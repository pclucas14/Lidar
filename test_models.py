from __future__ import print_function
import argparse
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from utils import * 
from models_2 import * 
from pointnet import * 
from datasets import PartDataset
from vae import * 

num_points = 64 ** 2 #2500 # 48 * 48
bottleneck_dim = 100 # 144
ext='_TEST_'

dataset = hkl.load('../prednet/kitti_data/X_test_lidar_raw_uniform.hkl').astype('float32') / 80.
train_loader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                           shuffle=True, num_workers=4, drop_last=True)
"""
Section 1 : VAEs
"""
'''
First Model : VAE trained with Chamfer Distance no kl: 
'''
vae_chamfer = VAE(4096, 256, ks=(2,3)) # ks should be overwritten by next line
vae_chamfer.decoder = PointGenPSG3(nz=256, bn=True, use_linear=True)
state_dict = torch.load('models/vae_lidar_epoch_60.pth')  
vae_chamfer.load_state_dict(state_dict)
vae_chamfer.cuda()
set_grad(vae_chamfer, False)


'''
Second Model : VAE trained with Chamfer Distance and KL with Lambda = 1e-7
'''
vae_chamfer_nokl = VAE(4096, 256, ks=(2,3)) # ks should be overwritten by next line
vae_chamfer_nokl.decoder = PointGen4096(nz=256, bn=False)
state_dict = torch.load('models/vae_lidar_epoch_45.pth')  
vae_chamfer_nokl.load_state_dict(state_dict)
vae_chamfer_nokl.cuda()
set_grad(vae_chamfer_nokl, False)

'''
Third Model : Adversarial VAE with lambda_rec, lambda_adv = 0.8, 0.2
'''
vaegan82 = VAE(4096, 100)
state_dict = torch.load('models/modelG_vaegan8.2_4096_81.pth')
vaegan82.decoder = PointGen4096(nz=100, ks=(2,2))
vaegan82.load_state_dict(state_dict)
vaegan82.cuda()
set_grad(vaegan82, False)

'''
Fourth Model : Adversarial VAE with lambda_rec, lambda_adv = .99, .01
'''
vaegan = VAE(4096, 100)
state_dict = torch.load('models/modelG_vaegan_4096_1051.pth')
vaegan.decoder = PointGen4096(nz=100, ks=(2,2))
vaegan.load_state_dict(state_dict)
vaegan.cuda()
set_grad(vaegan, False)

"""
Section 2 : GANs
"""

'''
First Model : 4 blocks of 1024 elements each
'''
gan_4096 = PointGen4096(nz=100, channels=[1024, 512, 256, 128, 3], num_blocks=4, ks=(2,2))
state_dict = torch.load('models/modelG_4096_31.pth') #31 seems ok, 21 maybe to play it safe
gan_4096.load_state_dict(state_dict)
gan_4096.cuda()
set_grad(gan_4096, False)


'''
Second Model : 16 miniblocks of 256 elements each
'''
gan_sb = PointGen4096(nz=100, channels=[256, 128, 64, 3], num_blocks=16, ks=(2,2))
state_dict = torch.load('models/modelG_16sb_21.pth') #31 seems ok, 21 maybe to play it safe
gan_sb.load_state_dict(state_dict)
gan_sb.cuda()
set_grad(gan_sb, False)

def show_samples(epoch):
    data_iter = iter(train_loader)
    iters = 0
    while iters < 10 : #len(train_loader)-1:
        iters += 1
        ''' gans '''
        inputv = Variable(torch.randn(32, 100)).cuda()
        fake_4096 = gan_4096(inputv)
        fake_sb = gan_sb(inputv)

        ''' vaes '''
        input = data_iter.next()
        inputv = Variable(input).cuda()
        fake_chamfer,      _, _ = vae_chamfer(inputv)
        fake_chamfer_nokl, _, _ = vae_chamfer_nokl(inputv)
        fake_vaegan,       _, _ = vaegan(inputv)
        fake_vaegan82,      _, _ = vaegan82(inputv)
        
        
        fake_4096 = fake_4096.cpu().data[0].numpy().transpose(1,0)
        fake_sb   = fake_sb.cpu().data[0].numpy().transpose(1,0)
        
        fake_chamfer      = fake_chamfer.cpu().data[0].numpy().transpose(1,0)
        fake_chamfer_nokl = fake_chamfer_nokl.cpu().data[0].numpy().transpose(1,0)
        fake_vaegan       = fake_vaegan.cpu().data[0].numpy().transpose(1,0)
        fake_vaegan82     = fake_vaegan82.cpu().data[0].numpy().transpose(1,0)
        real              = inputv.cpu().data.numpy()[0]


        hkl.dump(fake_4096, 'samples/' + 'gan_4096_' + str(iters) + '.hkl')
        hkl.dump(fake_sb, 'samples/' + 'gan_sb_' + str(iters) + '.hkl')
        
        hkl.dump(fake_chamfer, 'samples/' + 'vae_no_kl_' + str(iters) + '.hkl')
        hkl.dump(fake_chamfer_nokl, 'samples/' + 'vae_' + str(iters) + '.hkl')
        hkl.dump(fake_vaegan, 'samples/' + 'vaegan_' + str(iters) + '.hkl')
        hkl.dump(fake_vaegan82, 'samples/' + 'vaegan82_' + str(iters) + '.hkl')

        hkl.dump(real, 'samples/' + 'real_' + str(iters) + '.hkl')


print('starting training')
for epoch in range(1, 2):
    show_samples(epoch)
