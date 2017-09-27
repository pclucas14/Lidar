from __future__ import print_function
import argparse
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from utils import * 
from datasets import PartDataset
from vae_2d import * 

lambdas = [0, 1] # [.5, .5] #.999, .001]
lambda_mse, lambda_adv = lambdas
bottleneck_dim = 100 # 144
ext = '_vae_gan_'
use_wgan = True
critic_iters = 5 if use_wgan else 1
version = 'raw_lidar_wvae' if use_wgan else 'raw_lidar_lsvae'
epochs = 5000
USE_CARS = False
DEBUG = True

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if USE_CARS : 
    train_ds = PartDataset(root = 'pointGAN/shapenetcore_partanno_segmentation_benchmark_v0', class_choice = ['Car'], classification = True)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size,
                                          shuffle=True, num_workers=int(4), drop_last=True)

    test_ds = PartDataset(root = 'pointGAN/shapenetcore_partanno_segmentation_benchmark_v0', class_choice = ['Car'],classification = True, train = False)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size,
                                          shuffle=True, num_workers=int(4))
else : 
    dataset = hkl.load('../prednet/kitti_data/X_train_1d_bigger.hkl').astype('float32') / 80.
    dataset = dataset.transpose(0, 3, 1, 2)
    dataset = dataset[:, :3, :, :]
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                           shuffle=True, num_workers=4, drop_last=True)

num_batches = len(train_loader) 
netD = nnetD(1, nz=1, nc=1, lf=(3, 32))
model = VAE(f=(3,32), bottleneck_dim=100)
gen = model
#model.apply(weights_init)
#netD.apply(weights_init)

if args.cuda:
    model.cuda()
    netD.cuda()

if use_wgan : 
    optimizerD = optim.Adam(netD.parameters(), lr=1e-4)
    optimizerG = optim.Adam(model.parameters(), lr=1e-4)
else : 
    optimizerD = optim.RMSprop(netD.parameters(), lr=1e-4)
    optimizerG = optim.RMSprop(model.parameters(), lr=2e-4)
one = torch.FloatTensor([1]).cuda()
mone = torch.FloatTensor([-1.]).cuda() 

def set_grad(model, mode):
    for p in model.parameters():
        p.requires_grad = mode

def train(epoch):
    real_d, fake_d, fake_g, prior_g, prior_d, recon_loss, iters, rounds = [0] * 8
    data_iter = iter(train_loader)
    while iters < len(train_loader)-1:
        j = 0
        """ Update Discriminator Network """
        set_grad(netD, True); set_grad(model, False)
        rounds += 1
        while j < critic_iters and iters < len(train_loader)-1:
            optimizerD.zero_grad()
            j += 1; iters += 1
            # input, labels = data_iter.next()
            input = data_iter.next()
            inputv = Variable(input).cuda()
        
            # train with real data
            real_out = netD(inputv)
            real_d += real_out.mean().data[0]
            if use_wgan : 
                real_out.mean().backward(mone)
            else : 
                loss_real = torch.mean((real_out - 1.) ** 2)
            
            '''
            # train on prior
            prior = model.decoder(Variable(torch.randn(args.batch_size, 100)).cuda())
            prior_out, _ = netD(prior, activation=False)
            prior_d += prior_out.mean().data[0]
            if use_wgan : 
                prior_out.mean().backward(one)
            else : 
                loss_prior = torch.mean((prior_out - 0.) ** 2)
            '''    
            
            # train with fake data 
            fake, _, _ = model(inputv)
            fake_out = netD(fake)
            fake_d += fake_out.mean().data[0]
            # print('fake out : %s' % fake_out.mean())
            if use_wgan : 
                fake_out.mean().backward(one)
                grad_penalty = 10 * calc_gradient_penalty(netD, 
                                                     inputv.data, 
                                                     fake.data, 
                                                     batch_size=inputv.size(0))
                # print('grad penalty : %s' % grad_penalty.mean())
                grad_penalty.backward()
            else : 
                loss_fake = torch.mean((fake_out - 0.) ** 2) 
                loss = (loss_real + loss_fake) /2.# + loss_prior) / 3.
                loss.backward()
            
            optimizerD.step()
            
        """ Update Generator network """
        set_grad(netD, False); set_grad(model, True)
        optimizerG.zero_grad()
        iters += 1
        input = data_iter.next()
        inputv = Variable(input).cuda()
        
        fake, mu, logvar = model(inputv)
        
        fake_out, global_feat_fake = netD(fake, return_hidden=True)
        _       , global_feat_real = netD(inputv, return_hidden=True)
        fake_g += fake_out.mean().data[0]
        prior = model.decoder(Variable(torch.randn(args.batch_size, 100).normal_(0,1)).cuda())
        prior_out = netD(prior)
        if use_wgan : 
            prior_loss = prior_out.mean()
            prior_g += prior_out.data[0]
            prior_loss.backward()
            fake_out = lambda_adv * fake_out.mean()
            fake_out.backward(mone, retain_graph=True)
        else : 
            prior_loss = torch.mean((prior_out - 1.) ** 2)
            prior_g += prior_out.mean().data[0]
            prior_loss.backward()
            loss_adv = lambda_adv * torch.mean((fake_out - 1.) ** 2)
            loss_adv.backward(retain_graph=True)

        loss_mse = lambda_mse * torch.mean((global_feat_fake - global_feat_real) ** 2)
        
        # loss_mse = lambda_mse * loss_function(fake, inputv)#.mean()
        recon_loss += loss_mse.mean().data[0]
        loss_mse.backward()
        
        optimizerG.step()
        
            
    if epoch % 10 == 1 : 
        torch.save(netD.state_dict(), '%s/modelD%s%d.pth' % ('models', ext, epoch))
        torch.save(model.state_dict(), '%s/modelG%s%d.pth' % ('models', ext, epoch))
    
    # save sample
    print(str(epoch) + ' ' + str(ext))
    print(lambdas)
    print("real out D %.5f" % (real_d / (rounds)))
    print("fake out D %.5f" % (fake_d / (rounds)))
    # print("prior out D %.5f" % (prior_d / (100)))
    print("prior out G %.5f" % (prior_g / (rounds)))
    print("fake out G %.5f" % (fake_g / (rounds/critic_iters)))
    print("recon loss %.5f\n" % (recon_loss/rounds))
    real_d, fake_d, fake_g, prior_g, prior_d = [0.] * 5
    rounds, recon_loss = [0] * 2
    fake_sample = fake[0].permute(1,2,0).contiguous().cpu().data.numpy()
    real_sample = inputv[0].permute(1,2,0).contiguous().cpu().data.numpy()
    fake_sample = oned_to_threed(fake_sample).reshape(-1, 3)  
    real_sample = oned_to_threed(real_sample).reshape(-1, 3)  
    hkl.dump(fake_sample, 'clouds/fake' + ext + str(epoch) + '.hkl')
    hkl.dump(real_sample, 'clouds/real' + ext + str(epoch) + '.hkl')
    #hkl.dump(prior_sample, 'clouds/prior' + ext + str(epoch) + '.hkl')
    

print('starting training')
for epoch in range(1, epochs):
    train(epoch)
