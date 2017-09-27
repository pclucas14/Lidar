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
from pointnet import PointGenPSG3, PointGen4096 

class VAE(nn.Module):
    def __init__(self, num_points, bottleneck_dim, ks=(2,2)):
        super(VAE, self).__init__()
        # self.encoder = PC_Disc((args.batch_size, num_points, 3)) 
        self.encoder = PointNetfeat(num_points=num_points, global_dim=1024)
        self.fc_mu = nn.Linear(1024, bottleneck_dim)
        self.fc_sigma = nn.Linear(1024, bottleneck_dim)
        self.decoder = PointGen4096(nz=bottleneck_dim, ks=ks)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        x = x.transpose(2,1)
        enc = self.encoder(x)
        return self.fc_mu(enc), self.fc_sigma(enc)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if args.cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        dec = self.decoder(z)
        return dec

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar

def loss_function(recon_x, x, mu=0, logvar=0, epoch=0, lambda_kl=0):
    # take Chamfer Distance 95% of the time
    point_set_loss = (Chamfer_Dist(recon_x, x, gpus=GPUS)).mean()
    lambda_kl = 1e-7 if epoch > 5 else 0
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    return lambda_kl * KLD + point_set_loss


parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=5, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
GPUS = None


if __name__ == '__main__' : 
    num_points = 64 ** 2# 48 * 48
    base = 4
    ext='_vae_4096_kl_'
    bottleneck_dim = 256 # 144# 256#144 # 312
    lambda_kl =1e-7
    DEBUG = False


    model = VAE(num_points, bottleneck_dim)
    if args.cuda:
        model.cuda()

    #dataset = hkl.load('data.hkl').astype('float32')
    # dataset = hkl.load('../prednet/kitti_data/X_train_lidar_raw.hkl').astype('float32') / 80.
    dataset = hkl.load('../prednet/kitti_data/X_train_lidar_raw_uniform.hkl').astype('float32') / 80.
    # dataset = dataset[:, :num_points, :]

    split = int(0.92 * dataset.shape[0])
    d_train, d_test = dataset[:split], dataset[split:]

    kwargs = {'num_workers': 2, 'pin_memory': True, 'drop_last' : True}
    train_loader = torch.utils.data.DataLoader(d_train, 
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(d_test,
        batch_size=args.batch_size, shuffle=True, **kwargs)



    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    def train(epoch):
        model.train()
        train_loss = 0
        reps = 0
        for batch_idx, data in enumerate(train_loader):
            # data = torch.from_numpy(data)
            data = Variable(data)
            if args.cuda:
                data = data.cuda()
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar, epoch)
            loss.backward()
            train_loss += loss.data[0]
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                # save sample
                fake_sample = recon_batch.cpu().data[0].numpy().transpose(1,0)
                real_sample = data.cpu().data.numpy()[0]
                hkl.dump(fake_sample, 'clouds/fake' + ext + str(epoch) + '.hkl')
                hkl.dump(real_sample, 'clouds/real' + ext + str(epoch) + '.hkl')
                
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.10f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.data[0] / len(data)))

        print('====> Epoch: {} Average loss: {:.10f}'.format(
              epoch, train_loss / len(train_loader.dataset)))



    def test(epoch):
        model.eval()
        test_loss = 0
        for data in test_loader:
            if args.cuda:
                data = data.cuda()
            data = Variable(data, volatile=True)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar, 0).data[0]

        test_loss /= len(test_loader.dataset)
        print('====> Test set loss: {:.10f}'.format(test_loss))


    for epoch in range(0, 151):
        train(epoch)
        test(epoch)
        if epoch % 15 == 0 :
            torch.save(model.state_dict(), 'models/vae_lidar_epoch_%d.pth' % epoch)
            print('model saved')

    
    
