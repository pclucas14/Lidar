import argparse
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from utils import * 
from models_2 import * 

class VAE(nn.Module):
    def __init__(self, bottleneck_dim=10, f=(3, 16)):
        super(VAE, self).__init__()
        self.bottleneck_dim = bottleneck_dim
        self.encoder = nnetD(1, nz=1024, nc=1, lf=(1,1))
        self.fc_mu = nn.Conv2d(1024, bottleneck_dim, 5, 2) #nn.Linear(1024, bottleneck_dim)
        self.fc_sigma = nn.Conv2d(1024, bottleneck_dim, 5, 2)# nn.Linear(1024, bottleneck_dim)
        self.decoder = nnetG(1, nz=bottleneck_dim, nc=1, ff=f)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        enc = self.encoder(x)
        pdb.set_trace()
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
    # mask
    mask = ( x != 0. ).float()
    loss = torch.mean(mask * (recon_x - x) ** 2)
    # return loss
    # loss = torch.mean((recon_x - x) ** 2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    # print('KL : %s  loss : %s' % (KLD, loss))
    return lambda_kl * KLD + loss


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
    ext='_vae_kl3_'
    bottleneck_dim = 256 # 144# 256#144 # 312
    lambda_kl =1e-7
    DEBUG = False
    KEEP_PROB = 0.1

    model = VAE(f=(3,32))
    if args.cuda:
        model.cuda()

    dataset = hkl.load('../prednet/kitti_data/X_train_1d_bigger.hkl').astype('float32') / 80.
    print dataset.shape
    dataset = dataset.transpose(0, 3, 1, 2)
    dataset = dataset[:, :3, :, :]
    split = int(0.92 * dataset.shape[0])
    d_train, d_test = dataset[:split], dataset[split:]
    kwargs = {'num_workers': 2, 'pin_memory': True, 'drop_last' : True}
    train_loader = torch.utils.data.DataLoader(d_train, 
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(d_test,
        batch_size=args.batch_size, shuffle=True, **kwargs)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    def train(epoch):
        model.train()
        train_loss = 0
        reps = 0
        for batch_idx, data in enumerate(train_loader):
            # data = torch.from_numpy(data)
            '''
            og_data = Variable(data)
            mask = torch.ones(data.size())
            mask = mask * KEEP_PROB
            mask = torch.bernoulli(mask)
            data = data * mask
            data = data + torch.ones(data.size()).normal_(0, .2) #first was .2i, then 1
            '''
            data = Variable(data)
            # TODO:  remove this for partial output
            # data = og_data
            if args.cuda:
                data = data.cuda()
                # og_data = og_data.cuda()
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            lambda_kl = 0 if epoch == -1 else 1e-5
            loss = loss_function(recon_batch, data, mu, logvar, epoch, lambda_kl)
            loss.backward()
            train_loss += loss.data[0]
            optimizer.step()


        # check sample from prior
        noise = Variable(torch.ones(data.size(0), model.bottleneck_dim).normal_(0,1).cuda())
        # pdb.set_trace()
        prior_sample = model.decoder(noise)
        prior_sample = prior_sample[0].permute(1,2,0).contiguous().cpu().data.numpy()
         
        # save sample
        fake_sample = recon_batch[0].permute(1,2,0).contiguous().cpu().data.numpy()
        real_sample = data[0].permute(1,2,0).contiguous().cpu().data.numpy()
        #og_sample   = og_data[0].permute(1,2,0).contiguous().cpu().data.numpy()
        fake_sample = oned_to_threed(fake_sample).reshape(-1, 3)  
        real_sample = oned_to_threed(real_sample).reshape(-1, 3)  
        prior_sample = oned_to_threed(prior_sample).reshape(-1, 3)
        #og_sample   = oned_to_threed(og_sample).reshape(-1, 3)  
        hkl.dump(fake_sample, 'clouds/fake' + ext + str(epoch) + '.hkl')
        hkl.dump(real_sample, 'clouds/real' + ext + str(epoch) + '.hkl')
        hkl.dump(prior_sample, 'clouds/prior' + ext + str(epoch) + '.hkl')
        #hkl.dump(og_sample, 'clouds/og' + ext + str(epoch) + '.hkl')
        
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
            torch.save(model.state_dict(), 'models/' + ext + str(epoch) + '.pth')
            print('model saved')

    
    
