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

ext = '_gantestS_'
use_wgan = False
critic_iters = 5 if use_wgan else 1
epochs = 5000
USE_CARS = False
LOAD = False
one_d = True

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
netD = nnetD(1, ndf=64, nc=1, lf=(3,32))
model = nnetG(1, ngf=64, nc=1, ff=(3,32))
gen = model
print(netD); print(model)

if LOAD :
    pdb.set_trace()
    strG = ('%s/modelG%s%d.pth' % ('models', '_120_', 51))
    gen.load_state_dict(torch.load(strG))
    strD = ('%s/modelD%s%d.pth' % ('models', '_120_', 51))
    netD.load_state_dict(torch.load(strD))
else :     # for block in model.blocks : block.main.apply(weights_init)
    model.apply(weights_init)
    netD.apply(weights_init)

if args.cuda:
    model.cuda()
    netD.cuda()

optimizerD = optim.RMSprop(netD.parameters(), lr=1e-4)
optimizerG = optim.RMSprop(model.parameters(), lr=4e-4)

one = torch.FloatTensor([1]).cuda()
noise = torch.cuda.FloatTensor(args.batch_size, 100)
mone = one * -1
def train(epoch):
    data_iter = iter(train_loader)
    real_d, fake_d, fake_g, iters= [0] * 4
    while iters < len(train_loader):
        j = 0
        """ Update Discriminator Network """
        while j < critic_iters and iters < len(train_loader):
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
            
            # train with fake data 
            noisev = Variable(torch.randn(args.batch_size, 100)).cuda()
            fake = model(noisev)
            fake_out = netD(fake)
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
        optimizerG.zero_grad()
        noisev = Variable(torch.randn(args.batch_size, 100)).cuda()
        fake = model(noisev)
        fake_out = netD(fake)
        fake_g += fake_out.mean().data[0]
        if use_wgan : 
            fake_out.mean().backward(mone)
        else : 
            loss = torch.mean((fake_out - 1.) ** 2)
            loss.backward()
        
        optimizerG.step()

    mod = iters
    
    if epoch % 3 == 1 : #(iters + 1)  % 20 < critic_iters:
        print(str(epoch) + ' ' + str(ext))
        print("real out D %.5f" % (real_d / (mod)))
        print("fake out D %.5f" % (fake_d / (mod)))
        print("fake out G %.5f" % (fake_g / (mod / critic_iters)))
        real_d, fake_g, fake_d = [0] * 3
        # save sample
        if one_d : 
            fake_sample = fake[0].permute(1,2,0).contiguous().cpu().data.numpy() #transpose(0,2)
            real_sample = inputv[0].permute(1,2,0).contiguous().cpu().data.numpy()
            fake_sample = oned_to_threed(fake_sample).reshape(-1, 3)
            real_sample = oned_to_threed(real_sample).reshape(-1, 3)
        else : 
            fake_sample = fake[0].transpose(0,2).contiguous().cpu().data.numpy().reshape(-1, 3)
            real_sample = inputv[0].transpose(0,2).contiguous().cpu().data.numpy().reshape(-1, 3)
        
        hkl.dump(fake_sample, 'clouds/fake' + ext + str(epoch) + '.hkl')
        hkl.dump(real_sample, 'clouds/real' + ext + str(epoch) + '.hkl')
    
    if epoch % 10 == 0 : 
        torch.save(netD.state_dict(), '%s/modelD%s%d.pth' % ('models', ext, epoch))
        torch.save(model.state_dict(), '%s/modelG%s%d.pth' % ('models', ext, epoch))

for epoch in range(1, epochs):
    train(epoch)
