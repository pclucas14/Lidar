import argparse
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from pydoc import locate
import tensorboardX
import sys

from utils import * 
from models import * 

# reproducibility is good
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

nb_samples = 200
out_dir = os.path.join(sys.argv[1], 'final_samples')
maybe_create_dir(out_dir)
save_test_dataset = True


with torch.no_grad():
    # 1 ) unconditional generation
    print('unconditional generation')
    model = load_model_from_file(sys.argv[1], epoch=int(sys.argv[2]), model='gen')[0]
    model = model.cuda()
    samples = []

    with torch.no_grad():
        try:
            for temp in [0.2, 0.5, 0.7, 1]:
                z_ = model.args.z_dim
                is_vae = True
                model.eval()
                out = model.sample(nb_samples=nb_samples)
                np.save(os.path.join(out_dir, 'uncond_{}'.format(temp)), out.cpu().data.numpy())
        except:
            z_ = 100
            noise = torch.cuda.FloatTensor(nb_samples, z_).normal_()
            out = model(noise)
            is_vae = False
            np.save(os.path.join(out_dir, 'uncond'), out.cpu().data.numpy())


    # 2) undonditional interpolation
    print('unconditional interpolation')
    noise_a = torch.cuda.FloatTensor(nb_samples, z_).normal_()
    noise_b = torch.cuda.FloatTensor(nb_samples, z_).normal_()

    alpha  = np.arange(10) / 10.
    noises = []
    for a in alpha:
        noises += [a * noise_a + (1 - a) * noise_b]

    out = []
    for noise in noises:
        noise = noise.cuda()
        if is_vae:
            out += [model.decode(noise)]
        else:
            out += [model(noise)]

    out = torch.stack(out, dim=1)
    for i, inter in enumerate(out[:100]):
        np.save(os.path.join(out_dir, 'undond_inter_%d' % i), inter.cpu().data.numpy())

    if not is_vae:
        exit()

    # 3) test set reconstruction
    print('test set reconstruction')
    dataset = np.load('../../lidar_generation/kitti_data/lidar_test.npz')[::2]
    dataset = preprocess(dataset).astype('float32')

    if save_test_dataset: 
        np.save(os.path.join(out_dir, 'test_set'), dataset)

    dataset_test = from_polar_np(dataset) if model.args.no_polar else dataset
    loader = iter(torch.utils.data.DataLoader(dataset_test, batch_size=100,
                        shuffle=True, num_workers=4, drop_last=False))

    real_data = next(loader).cuda()
    out = model(real_data)[0]
    np.save(os.path.join(out_dir, 'recon'), out.cpu().data.numpy())

    print('test set interpolation')
    aa, bb = real_data, next(loader).cuda()

    noise_a, noise_b = model.encode(aa).chunk(2, dim=1)[0], model.encode(bb).chunk(2, dim=1)[0]
    alpha  = np.arange(10) / 10.
    noises, out = [], []
    for a in alpha:
        noises += [a * noise_a + (1 - a) * noise_b]

    for noise in noises:
        noise = noise.cuda()
        out += [model.decode(noise)]

    out = torch.stack(out, dim=1)
    # add ground truth to saved tensors
    out = torch.cat([bb.unsqueeze(1), out, aa.unsqueeze(1)], dim=1)
    for i, inter in enumerate(out[:100]):
        np.save(os.path.join(out_dir, 'cond_inter_%d' % i), inter.cpu().data.numpy())
    
