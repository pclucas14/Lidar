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
import torch.nn.utils.spectral_norm as spectral_norm


# --------------------------------------------------------------------------
# Define rotation equivariant layers
# -------------------------------------------------------------------------- 
class round_conv2d(nn.Conv2d):
    def __init__(self, channels_in, channels_out, filter_size, stride=1, padding=(0,0), bias=True):
        if isinstance(padding, int):
            padding = (padding, padding)

        super().__init__(channels_in, channels_out, filter_size, stride=stride, padding=(padding[0], 0), bias=bias)
        self.padding_ = padding[1]

    def forward(self, x):
        # first, we pad the input
        input = x
        if self.padding_ > 0:
            x = torch.cat([x[:, :, :, -self.padding_:], x, x[:, :, :, :self.padding_]], dim=-1)
        out = super().forward(x)
        return out

class round_deconv2d(nn.ConvTranspose2d):
    def __init__(self, channels_in, channels_out, filter_size, stride=1, padding=(0,0), bias=True):
        if isinstance(padding, int):
            padding = (padding, padding)
        
        super().__init__(channels_in, channels_out, filter_size, stride=stride, padding=(padding[0], 0), bias=bias)
        
        self.padding_ = padding[1]

    def forward(self, x):
        input = x
        if self.padding_ > 0:
           x = torch.cat([x[:, :, :, -self.padding_:], x, x[:, :, :, 0:self.padding_]], dim=-1)
        out = super().forward(x)
        return out

'''
Round two of models
'''

class netG(nn.Module):
    def __init__(self, args, nz=100, ngf=64, nc=3, base=4, ff=(2,16)):
        super(netG, self).__init__()
        conv = round_deconv2d if args.use_round_conv else nn.ConvTranspose2d
        sn   = spectral_norm if args.use_spectral_norm else lambda x : x

        layers  = []
        layers += [sn(conv(nz, ngf * 8, ff, 1, 0, bias=False))]
        
        if args.use_selu:
            layers += [nn.SELU()]
        else: 
            layers += [nn.BatchNorm2d(ngf * 8)] 
            layers += [nn.ReLU(True)]

        layers += [sn(conv(ngf * 8, ngf * 4, (3,4), stride=2, padding=(0,1), bias=False))]

        if args.use_selu:
            layers += [nn.SELU()]
        else: 
            layers += [nn.BatchNorm2d(ngf * 4)] 
            layers += [nn.ReLU(True)]
        
        layers += [sn(conv(ngf * 4, ngf * 2, (4,4), stride=2, padding=(1,1), bias=False))]

        if args.use_selu:
            layers += [nn.SELU()]
        else: 
            layers += [nn.BatchNorm2d(ngf * 2)] 
            layers += [nn.ReLU(True)]
        
        layers += [sn(conv(ngf * 2, ngf * 1, (4,4), stride=2, padding=(1,1), bias=False))]

        if args.use_selu:
            layers += [nn.SELU()]
        else: 
            layers += [nn.BatchNorm2d(ngf * 1)] 
            layers += [nn.ReLU(True)]

        layers += [sn(conv(ngf, nc, 4, 2, 1, bias=False))]
        layers += [nn.Tanh()]

        self.main = nn.Sequential(*layers)

    def forward(self, input):
        if len(input.shape) == 2: 
            input = input.unsqueeze(-1).unsqueeze(-1)
        
        return self.main(input)


class netD(nn.Module):
    def __init__(self, args, ndf=64, nc=3, nz=1, lf=(2,16)):
        super(netD, self).__init__()
        self.encoder = True if nz > 1 else False
        
        conv = round_conv2d if args.use_round_conv else nn.Conv2d
        sn   = spectral_norm if args.use_spectral_norm else (lambda x : x)

        layers  = []
        layers += [sn(conv(nc, ndf, 4, 2, 1, bias=False))]
        
        if args.use_selu:
            layers += [nn.SELU()]
        else: 
            layers += [nn.LeakyReLU(0.2, inplace=True)]
        
        layers += [sn(conv(ndf, ndf * 2, 4, 2, 1, bias=False))]
        
        if args.use_selu:
            layers += [nn.SELU()]
        else: 
            layers += [nn.BatchNorm2d(ndf * 2)]
            layers += [nn.LeakyReLU(0.2, inplace=True)]
        
        layers += [sn(conv(ndf * 2, ndf * 4, 4, 2, 1, bias=False))]
        
        if args.use_selu:
            layers += [nn.SELU()]
        else: 
            layers += [nn.BatchNorm2d(ndf * 4)]
            layers += [nn.LeakyReLU(0.2, inplace=True)]
        
        layers += [sn(conv(ndf * 4, ndf * 8, (3,4), 2, (0,1), bias=False))]
        
        if args.use_selu:
            layers += [nn.SELU()]
        else: 
            layers += [nn.BatchNorm2d(ndf * 8)]
            layers += [nn.LeakyReLU(0.2, inplace=True)]

        self.main = nn.Sequential(*layers)
        
        self.out = sn(conv(ndf * 8, nz, lf, 1, 0, bias=False))

    def forward(self, input, return_hidden=False):
        if input.size(-1) == 3: 
            input = input.transpose(1, 3)
        
        output_tmp = self.main(input)
        output = self.out(output_tmp)
       
        if return_hidden:
            return output, output_tmp
        
        return output if self.encoder else output.view(-1, 1).squeeze(1) 


if __name__ == '__main__':
    noise = torch.FloatTensor(1, 100, 1, 1).normal_()
    gen = netG(1)
    fake = gen.main(noise)
    dis = netD(1)
    fake_out = dis(fake)
    import pdb; pdb.set_trace()
