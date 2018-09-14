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
from emd import EMD

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

EMD = EMD()


with torch.no_grad():

    # 1) load trained model
    model = load_model_from_file(sys.argv[1], epoch=int(sys.argv[2]), model='gen')[0]
    model = model.cuda()
    model.eval()
    
    # 2) load data
    print('test set reconstruction')
    dataset = np.load('../../lidar_generation/kitti_data/lidar_test.npz')
    dataset = preprocess(dataset).astype('float32')

    if save_test_dataset: 
        np.save(os.path.join(out_dir, 'test_set'), dataset)

    dataset_test = from_polar_np(dataset) if model.args.no_polar else dataset
    loader = iter(torch.utils.data.DataLoader(dataset_test, batch_size=dataset.shape[0],
                        shuffle=True, num_workers=4, drop_last=False))


    batch = next(loader)

    # regular reconstruction
    recon = model(batch)[0]
    
    emd_clean = EMD(recon, batch)
    import pdb; pdb.set_trace()

