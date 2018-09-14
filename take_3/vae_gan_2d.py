import argparse
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from pydoc import locate
import tensorboardX

from utils import * 
from models import * 

parser = argparse.ArgumentParser(description='VAE training of LiDAR')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--use_selu', type=int, default=0, help='replaces batch_norm + act with SELU')
parser.add_argument('--use_spectral_norm', type=int, default=0)
parser.add_argument('--use_round_conv', type=int, default=0)
parser.add_argument('--base_dir', type=str, default='runs/test')
parser.add_argument('--no_polar', type=int, default=0)
parser.add_argument('--optim',  type=str, default='Adam')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--z_dim', type=int, default=128)
parser.add_argument('--iaf', type=int, default=0)
parser.add_argument('--autoencoder', type=int, default=0)
parser.add_argument('--atlas_baseline', type=int, default=0)
parser.add_argument('--kl_warmup_epochs', type=int, default=150)
parser.add_argument('--lambda_recon', type=float, default=0.5)

args = parser.parse_args()
maybe_create_dir(args.base_dir)
print_and_save_args(args, args.base_dir)

# reproducibility is good
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

# construct model and ship to GPU
model = VAE(args).cuda()

# load discriminator for loss computation
path = '/scratch/lpagec/lidar_generation/gan_classic/Conv0_Selu0_SN0_Loss0_BS128_OPTadamGLR:0.0002_DLR:0.0001XYZ:0'
dis = load_model_from_file(path, epoch=999)[0].cuda()

# Logging
maybe_create_dir(os.path.join(args.base_dir, 'samples'))
maybe_create_dir(os.path.join(args.base_dir, 'models'))
writer = tensorboardX.SummaryWriter(log_dir=os.path.join(args.base_dir, 'TB'))
writes = 0
ns     = 16

# dataset preprocessing
dataset = np.load('../../lidar_generation/kitti_data/lidar.npz') 
dataset = preprocess(dataset).astype('float32')
dataset_train = from_polar_np(dataset) if args.no_polar else dataset

dataset = np.load('../../lidar_generation/kitti_data/lidar_val.npz') 
dataset = preprocess(dataset).astype('float32')
dataset_val = from_polar_np(dataset) if args.no_polar else dataset

train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size,
                    shuffle=True, num_workers=4, drop_last=True)

val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size,
                    shuffle=True, num_workers=4, drop_last=False)

print(model)
model.apply(weights_init)
optim = optim.Adam(model.parameters(), lr=args.lr) 

# construction reconstruction loss function
def loss_fn(a, b):
    int_a = dis(a, return_hidden=True)[1]
    int_b = dis(b, return_hidden=True)[1]
    adv_loss = (int_a - int_b).abs().sum(-1).sum(-1).sum(-1) / sum([x for x in int_a.shape[1:]])
    mse_loss = (a - b).abs().sum(-1).sum(-1).sum(-1) / sum([x for x in a.shape[1:]])
    return args.lambda_recon * mse_loss + (1 - args.lambda_recon) * adv_loss
    


# VAE training
# ------------------------------------------------------------------------------------------------
for epoch in range(1000):
    print('epoch %s' % epoch)
    model.train()
    loss_, elbo_, kl_cost_, kl_obj_, recon_ = [[] for _ in range(5)]

    for i, img in enumerate(train_loader):
        img = img.cuda()
        recon, kl_cost = model(img)

        loss_recon = loss_fn(recon, img)

        kl_obj  =  min(1, float(epoch) / args.kl_warmup_epochs) * torch.clamp(kl_cost, min=5)

        loss = (kl_obj + loss_recon).mean(dim=0)

        elbo = (kl_cost + loss_recon).mean(dim=0)

        loss_    += [loss.item()]
        elbo_    += [elbo.item()]
        kl_cost_ += [kl_cost.mean(dim=0).item()]
        kl_obj_  += [kl_obj.mean(dim=0).item()]
        recon_   += [loss_recon.mean(dim=0).item()]

        optim.zero_grad()
        loss.backward()
        optim.step()

    writes += 1
    mn = lambda x : np.mean(x)
    print_and_log_scalar(writer, 'train/loss', mn(loss_), writes)
    print_and_log_scalar(writer, 'train/elbo', mn(elbo_), writes)
    print_and_log_scalar(writer, 'train/kl_cost_', mn(kl_cost_), writes)
    print_and_log_scalar(writer, 'train/kl_obj_', mn(kl_obj_), writes)
    print_and_log_scalar(writer, 'train/recon', mn(recon_), writes)
    loss_, elbo_, kl_cost_, kl_obj_, recon_ = [[] for _ in range(5)]
        
    # save some training reconstructions
    recon = recon[:ns].cpu().data.numpy()
    with open(os.path.join(args.base_dir, 'samples/train_{}.npz'.format(epoch)), 'wb') as f: 
        np.save(f, recon)

    # log_point_clouds(writer, recon[:ns], 'train_recon', step=writes)
    print('saved training reconstructions')
    
    
    # Testing loop
    # ----------------------------------------------------------------------

    loss_, elbo_, kl_cost_, kl_obj_, recon_ = [[] for _ in range(5)]
    with torch.no_grad():
        model.eval()
        if epoch % 1 == 0:
            print('test set evaluation')
            for i, img in enumerate(val_loader):
                img = img.cuda()
                recon, kl_cost = model(img)
            
                loss_recon = loss_fn(recon, img)
        
                kl_obj  =  min(1, float(epoch) / args.kl_warmup_epochs) * torch.clamp(kl_cost, min=5)

                loss = (kl_obj + loss_recon).mean(dim=0)

                elbo = (kl_cost + loss_recon).mean(dim=0)

                loss_    += [loss.item()]
                elbo_    += [elbo.item()]
                kl_cost_ += [kl_cost.mean(dim=0).item()]
                kl_obj_  += [kl_obj.mean(dim=0).item()]
                recon_   += [loss_recon.mean(dim=0).item()]

                # if epoch % 5 != 0 and i > 5 : break

            print_and_log_scalar(writer, 'valid/loss', mn(loss_), writes)
            print_and_log_scalar(writer, 'valid/elbo', mn(elbo_), writes)
            print_and_log_scalar(writer, 'valid/kl_cost_', mn(kl_cost_), writes)
            print_and_log_scalar(writer, 'valid/kl_obj_', mn(kl_obj_), writes)
            print_and_log_scalar(writer, 'valid/recon', mn(recon_), writes)
            loss_, elbo_, kl_cost_, kl_obj_, recon_ = [[] for _ in range(5)]

            with open(os.path.join(args.base_dir, 'samples/test_{}.npz'.format(epoch)), 'wb') as f: 
                recon = recon[:ns].cpu().data.numpy()
                np.save(f, recon)
                print('saved test recons')

           
            sample = model.sample()
            with open(os.path.join(args.base_dir, 'samples/sample_{}.npz'.format(epoch)), 'wb') as f: 
                sample = sample.cpu().data.numpy()
                np.save(f, recon)
            
            #log_point_clouds(writer, sample, 'vae_samples', step=writes)
            print('saved model samples')

            if epoch == 0: 
                with open(os.path.join(args.base_dir, 'samples/real.npz'), 'wb') as f: 
                    img = img.cpu().data.numpy()
                    np.save(f, img)
                
                # log_point_clouds(writer, img[:ns], 'real_lidar', step=writes)
                print('saved real LiDAR')

    if (epoch + 1) % 10 == 0 :
        torch.save(model.state_dict(), os.path.join(args.base_dir, 'models/gen_{}.pth'.format(epoch)))
