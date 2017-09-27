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
lambdas = [.99, .01]
lambda_mse, lambda_adv = lambdas
base = 3
bottleneck_dim = 100 # 144
ext = '_vae0_'
use_wgan = False
critic_iters = 5 if use_wgan else 3
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
    dataset = hkl.load('../prednet/kitti_data/X_train_lidar_raw_uniform.hkl').astype('float32') / 80.
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                           shuffle=True, num_workers=4, drop_last=True)

num_batches = len(train_loader) 
netD = PointNetCls(k=2, num_points = num_points)
model = VAE(num_points, bottleneck_dim)
gen = model
model.encoder.apply(weights_init)
for block in model.decoder.blocks : block.main.apply(weights_init)
netD.apply(weights_init)

if args.cuda:
    model.cuda()
    netD.cuda()

optimizerD = optim.RMSprop(netD.parameters(), lr=1e-4)
optimizerG = optim.RMSprop(model.parameters(), lr=2e-4)
one = torch.FloatTensor([1]).cuda()
mone = torch.FloatTensor([-1.]).cuda() 

def set_grad(model, mode):
    for p in model.parameters():
        p.requires_grad = mode

def train(epoch):
    real_d, fake_d, fake_g, prior_g, recon_loss, iters, rounds = [0] * 7
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
            real_out, _ = netD(inputv.transpose(2,1), activation=False)
            real_d += real_out.mean().data[0]
            if use_wgan : 
                real_out.mean().backward(mone)
            else : 
                loss_real = torch.mean((real_out - 1.) ** 2)
            
            # train with fake data 
            fake, _, _ = model(inputv)
            fake_out, _ = netD(fake, activation=False)
            fake_d += fake_out.mean().data[0]
            if use_wgan : 
                fake_out.mean().backward(one)
                grad_penalty = 10 * calc_gradient_penalty(netD, 
                                                     inputv.transpose(2,1).data, 
                                                     fake.data, 
                                                     batch_size=inputv.size(0))
                grad_penalty.backward()
            else : 
                loss_fake = torch.mean((fake_out - 0.) ** 2) 
                loss = (loss_real + loss_fake) / 2.
                loss.backward()
            
            optimizerD.step()
            
        """ Update Generator network """
        set_grad(netD, False); set_grad(model, True)
        optimizerG.zero_grad()
        iters += 1
        input = data_iter.next()
        inputv = Variable(input).cuda()
        
        fake, _, _ = model(inputv)
        fake_out, global_feat_fake = netD(fake, activation=False)
        _       , global_feat_real = netD(inputv.transpose(2,1), activation=False)
        fake_g += fake_out.mean().data[0]
        prior = model.decoder(Variable(torch.randn(args.batch_size, 100)).cuda())
        prior_out, _ = netD(prior, activation=False)
        if use_wgan : 
            prior_loss = prior_out.mean()
            prior_g += prior_out.data[0]
            prior_loss.backward()
            fake_out = lambda_adv * fake_out.mean()
            fake_out.backward(mone, retain_graph=True)
        else : 
            #prior_loss = torch.mean((prior_out - 1.) ** 2)
            #prior_g += prior_out.mean().data[0]
            #prior_loss.backward()
            loss_adv = lambda_adv * torch.mean((fake_out - 1.) ** 2)
            loss_adv.backward(retain_graph=True)

        loss_mse = lambda_mse * torch.mean((global_feat_fake - global_feat_real) ** 2)
        # loss_mse = lambda_mse * loss_function(fake, inputv)
        recon_loss += loss_mse.mean().data[0]
        loss_mse.backward()
        
        optimizerG.step()
        
        if (iters + 1) % 100 < (critic_iters+1) : 
            # save sample
            print(str(epoch) + ' ' + str(ext))
            print(lambdas)
            print("real out D %.5f" % (real_d / (100)))
            print("fake out D %.5f" % (fake_d / (100)))
            print("prior out G %.5f" % (prior_g / (100)))
            print("fake out G %.5f" % (fake_g / (100/critic_iters)))
            print("recon loss %.5f\n" % (recon_loss))
            real_d, fake_d, fake_g, prior_g = [0.] * 4
            rounds, recon_loss = [0] * 2
            fake_sample = fake.cpu().data[0].numpy().transpose(1,0)
            prior_sample = prior.cpu().data[0].numpy().transpose(1,0)
            real_sample = inputv.cpu().data.numpy()[0]
            hkl.dump(fake_sample, 'clouds/fake' + ext + str(epoch) + '.hkl')
            hkl.dump(real_sample, 'clouds/real' + ext + str(epoch) + '.hkl')
            hkl.dump(prior_sample, 'clouds/prior' + ext + str(epoch) + '.hkl')
            
        if epoch % 10 == 1 : 
            torch.save(netD.state_dict(), '%s/modelD%s%d.pth' % ('models', ext, epoch))
            torch.save(model.state_dict(), '%s/modelG%s%d.pth' % ('models', ext, epoch))

    print(epoch)
    print("real out D %.5f" % (real_d / (iters)))
    print("fake out D %.5f" % (fake_d / (iters)))
    print("fake out G %.5f" % (fake_g / (iters)))# / critic_iters)))
    

    

print('starting training')
for epoch in range(1, epochs):
    train(epoch)
