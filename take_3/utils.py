import numpy as np
import torch
import torch.nn.functional as F
import os

# -------------------------------------------------------------------------
# Handy Utilities
# -------------------------------------------------------------------------
def to_polar(velo):
    if len(velo.shape) == 4:
        velo = velo.transpose(1, 2, 3, 0)
    # assumes r x n/r x (3,4) velo
    dist = np.sqrt(velo[:, :, 0] ** 2 + velo[:, :, 1] ** 2)
    # theta = np.arctan2(velo[:, 1], velo[:, 0])
    out = np.stack([dist, velo[:, :, 2]], axis=2)
    if len(velo.shape) == 4: 
        out = out.transpose(3, 0, 1, 2)
    return out

def from_polar(velo):
    angles = np.linspace(0, np.pi * 2, velo.shape[-1])
    dist, z = velo[:, 0], velo[:, 1]
    x = torch.Tensor(np.cos(angles)).cuda().unsqueeze(0).unsqueeze(0) * dist
    y = torch.Tensor(np.sin(angles)).cuda().unsqueeze(0).unsqueeze(0) * dist
    out = torch.stack([x,y,z], dim=1)
    return out

def from_polar_np(velo):
    angles = np.linspace(0, np.pi * 2, velo.shape[-1])
    dist, z = velo[:, 0], velo[:, 1]
    x = np.cos(angles) * dist
    y = np.sin(angles) * dist
    out = np.stack([x,y,z], axis=1)
    return out

def print_and_log_scalar(writer, name, value, write_no, end_token=''):
    if isinstance(value, list):
        if len(value) == 0: return 
        value = torch.mean(torch.stack(value))
    zeros = 40 - len(name) 
    name += ' ' * zeros
    print('{} @ write {} = {:.4f}{}'.format(name, write_no, value, end_token))
    writer.add_scalar(name, value, write_no)

def log_point_clouds(writer, data, name, step):
    if len(data.shape) == 3:
        data = [data]
    
    out = np.stack([from_polar(x.transpose(1, 2, 0)) for x in \
            data.cpu().data.numpy()])
    out = torch.tensor(out).float()

    for i, cloud in enumerate(out):
        cloud = cloud.view(-1, 3)
        writer.add_embedding(cloud, tag=name + '_%d' % i, global_step=step)

def print_and_save_args(args, path):
    print(args)
    # let's save the args as json to enable easy loading
    import json
    with open(os.path.join(path, 'args.json'), 'w') as f: 
        json.dump(vars(args), f)

def maybe_create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def remove_zeros(pc):
    xx = torch.cuda.FloatTensor(pc)
    if xx.dim() == 3: 
        xx = xx.unsqueeze(0)

    while (xx[:, 0] == 0).sum() > 0 : 
        mask = (xx[:, 0] == 0).unsqueeze(1).float()
        out_a = F.max_pool2d(xx[:, 0], 5, padding=2, stride=1)
        out_b = -F.max_pool2d(-xx[:, 1], (5, 5), padding=(2, 2), stride=1)
        #out_b_ = (xx[:, 1]).min(dim=-1, keepdim=True)[0].expand_as(out_b)
        #out_b = torch.cat([out_b_[:, :10], out_b[:, 10:]], dim=1)
        out_b = out_b.expand_as(out_a)
        out = torch.stack([out_a, out_b], dim=1)
        mask = (xx[:, 0] == 0).unsqueeze(1)
        mask = mask.float()
        xx = xx * (1 - mask) + (mask) * out

    return xx.cpu().data.numpy()


def preprocess(dataset):
    # remove outliers 
    #min_a, max_a = np.percentile(dataset[:, :, :, [0]], 1), np.percentile(dataset[:, :, :, [0]], 99)
    #min_b, max_b = np.percentile(dataset[:, :, :, [1]], 1), np.percentile(dataset[:, :, :, [1]], 99)
    #min_c, max_c = np.percentile(dataset[:, :, :, [2]], 1), np.percentile(dataset[:, :, :, [2]], 99)
    min_a, max_a = -41.1245002746582,   36.833248138427734
    min_b, max_b = -25.833599090576172, 30.474000930786133
    min_c, max_c = -2.3989999294281006, 0.7383332848548889
    dataset = dataset[::25, 5:45]

    mask = np.maximum(dataset[:, :, :, 0] < min_a, dataset[:, :, :, 0] > max_a)
    mask = np.maximum(mask, np.maximum(dataset[:, :, :, 1] < min_b, dataset[:, :, :, 1] > max_b))
    mask = np.maximum(mask, np.maximum(dataset[:, :, :, 2] < min_c, dataset[:, :, :, 2] > max_c))
    
    dist = dataset[:, :, :, 0] ** 2 + dataset[:, :, :, 1] ** 2
    mask = np.maximum(mask, dist < 7)

    dataset = dataset * (1 - np.expand_dims(mask, -1))
    dataset /= np.absolute(dataset).max()

    dataset = to_polar(dataset).transpose(0, 3, 1, 2)
    previous = (dataset[:, 0] == 0).sum()

    for i in range(dataset.shape[0]):
        print('processing {}/{}'.format(i, dataset.shape[0]))
        pp = remove_zeros(dataset[i]).squeeze(0)
        dataset[i] = pp

    return dataset[:, :, :, ::2]

def show_pc(velo):
    import mayavi.mlab
    #fig = mayavi.mlab.figure(bgcolor=(0, 0, 0), size=(640, 360))
    if len(velo.shape) == 3: 
        if velo.shape[2] != 3:
            velo = velo.transpose(1,2,0)

        velo = velo.reshape((-1, 3))

    nodes = mayavi.mlab.points3d(
        velo[:, 0],   # x
        velo[:, 1],   # y
        velo[:, 2],   # z
        scale_factor=0.005, #0.022,     # scale of the points
    )
    nodes.glyph.scale_mode = 'scale_by_vector'
    color = (velo[:, 2] - velo[:, 2].min()) / (velo[:, 2].max() - velo[:, 2].min())
    nodes.mlab_source.dataset.point_data.scalars = color
    print('showing pc')
    mayavi.mlab.show()

if __name__ == '__main__':
    import hickle as hkl
    import sys
    xx = hkl.load('clouds/real%s.hkl' % sys.argv[1])
    import pdb; pdb.set_trace()
    out = from_polar_np(np.expand_dims(xx, 0))
    outp = remove_zeros(xx)
    outp = from_polar_np(outp)
    show_pc(out[0])
    show_pc(outp[0])
    x = 1
