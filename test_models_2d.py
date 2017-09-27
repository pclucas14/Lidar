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
from vae_2d import * 

num_points = 64 ** 2 #2500 # 48 * 48
bottleneck_dim = 100 # 144
ext='_TEST_'
KEEP_PROB = .05

dataset = hkl.load('../prednet/kitti_data/X_test_1d_512_.hkl').astype('float32') / 80.
dataset = dataset.transpose(0, 3, 1, 2)
dataset = dataset[:, :3, :, :]
print dataset.shape
train_loader = torch.utils.data.DataLoader(dataset, batch_size=16,
                                           shuffle=False, num_workers=4, drop_last=False)
"""
Section 1 : VAEs
"""
'''
First Model : VAE with a bottleneck of 100 
'''
vaes = []
vae = VAE(bottleneck_dim=100, f=(3,32))
state_dict = torch.load('models/_vae_pie_100_512_15.pth')  
vae.load_state_dict(state_dict)
vae.cuda()
set_grad(vae, False)
vaes.append(vae)

vae = VAE(bottleneck_dim=100, f=(3,32))
state_dict = torch.load('models/_va_keep_.5_15.pth')  
vae.load_state_dict(state_dict)
vae.cuda()
set_grad(vae, False)
vaes.append(vae)


vae = VAE(bottleneck_dim=100, f=(3,32))
state_dict = torch.load('models/_va_keep_.05_15.pth')  
vae.load_state_dict(state_dict)
vae.cuda()
set_grad(vae, False)
vaes.append(vae)

vae = VAE(bottleneck_dim=100, f=(3,32))
state_dict = torch.load('models/_va_denoising_15.pth')  
vae.load_state_dict(state_dict)
vae.cuda()
set_grad(vae, False)
vaes.append(vae)

vae = VAE(bottleneck_dim=100, f=(3,32))
state_dict = torch.load('models/_va_denoising_115.pth')  
vae.load_state_dict(state_dict)
vae.cuda()
set_grad(vae, False)
vaes.append(vae)


vae = VAE(bottleneck_dim=100, f=(3,32))
state_dict = torch.load('models/_va_both15.pth')  
vae.load_state_dict(state_dict)
vae.cuda()
set_grad(vae, False)
vaes.append(vae)
"""
'''
Second Model : VAE GAN 
'''
vae_40 = VAE(bottleneck_dim=100, f=(3,32))
state_dict = torch.load('models/_va_both60.pth')  
vae_40.load_state_dict(state_dict)
vae_40.cuda()
set_grad(vae_40, False)
"""
"""
Section 2 : GANs
"""
"""
'''
First Model :Good'ol DCGAN with LSGAN loss
'''
gan = nnetG(1, nc=1, ff=(3,32))
state_dict = torch.load('models/modelG_gantestS_450.pth') #31 seems ok, 21 maybe to play it safe
gan.load_state_dict(state_dict)
gan.cuda()
set_grad(gan, False)
"""

def show_samples(epoch):
    data_iter = iter(train_loader)
    iters = 0
    losses = [0.] * 6
    while iters < len(train_loader)-1:
        iters += 1
        ''' gans '''
        #inputv = Variable(torch.randn(32, 100)).cuda()
        #fake_gan = gan(inputv)
      
        ''' vaes '''
        inputv = data_iter.next()
        input = inputv
        inputv = Variable(inputv).cuda()
        #fake_vae_40, _, _ = vae_40(inputv)
        # fake_vae_gan, _, _ = vae_gan(inputv)
        inputs = []
        inputs.append(inputv)

        """ mask input """
        # og_data = Variable(input).cuda()
        mask = torch.ones(input.size())
        mask = mask * .5
        mask = torch.bernoulli(mask)
        data = input * mask
        data_f = Variable(data).cuda()
        inputs.append(data_f)

        mask = torch.ones(input.size())
        mask = mask * .05
        mask = torch.bernoulli(mask)
        data = input * mask
        data_pf = Variable(data).cuda()
        inputs.append(data_pf)

        # noise 
        data = input + torch.ones(data.size()).normal_(0, .2)
        inputs.append(Variable(data).cuda())
        data = input + torch.ones(data.size()).normal_(0, 1)
        inputs.append(Variable(data).cuda())
        data = data_pf.cpu().data + torch.ones(data.size()).normal_(0, .2)
        inputs.append(Variable(data).cuda())
        #inputv = Variable(data).cuda()
        #fake_vae_40, _, _ = vae_40(inputv) 
        for i in range(len(vaes)):
            out,_,_ = vaes[i](inputs[i])
            loss = torch.mean((out - inputv) ** 2) / 2.
            losses[i] += loss.cpu().data[0]

        '''
        fake_vae_40 = oned_to_threed(fake_vae_40[0].permute(1,2,0).contiguous().cpu().data.numpy()).reshape(-1,3)
        # fake_vae_gan = oned_to_threed(fake_vae_gan[0].permute(1,2,0).contiguous().cpu().data.numpy()).reshape(-1,3)
        fake_gan    = oned_to_threed(fake_gan[0].permute(1,2,0).contiguous().cpu().data.numpy()).reshape(-1,3)
        real        = oned_to_threed(inputv[0].permute(1,2,0).contiguous().cpu().data.numpy()).reshape(-1,3)
        og        = oned_to_threed(og_data[0].permute(1,2,0).contiguous().cpu().data.numpy()).reshape(-1,3)
        

        hkl.dump(fake_vae_40, 'samples2/' + 'vae_.05.2_' + str(iters) + '.hkl')
        hkl.dump(og, 'samples2/' + 'og_' + str(iters) + '.hkl')
        # hkl.dump(fake_vae_gan, 'samples/' + 'vae_gan_' + str(iters) + '.hkl')
        # hkl.dump(fake_gan, 'samples/' + 'gan_' + str(iters) + '.hkl')
        hkl.dump(real, 'samples2/' + 'real_' + str(iters) + '.hkl')
        '''
    print(losses)
    print(len(dataset.shape))
    pdb.set_trace()
print('starting training')
for epoch in range(1, 2):
    show_samples(epoch)
