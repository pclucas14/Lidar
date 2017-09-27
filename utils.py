import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import torch.nn as nn
import torchvision
import numpy as np
from numpy import unravel_index
import matplotlib.pyplot as plt
import torch.optim as optim
import pdb
import hickle as hkl
import torch.autograd as autograd
from PIL import Image
import ot

points = 0.2
points_step = int(1. / points)
point_size = 0.01 * (1. / points)
axes_limits = [[-25, 25], [-18, 18], [-1, 1]]
axes_str = ['X', 'Y', 'Z']


def lin_inter(a, b, num_pts=100):
    prop = np.stack([np.arange(0, 1, 1./num_pts)] * 2, axis=1)
    inter = a * prop + b * prop[::-1]
    return inter


def oned_to_threed(velo):
    heights = np.arange(-.02, .02, .04 / velo.shape[0])
    # velo has shape 60, 512, 1
    out = np.zeros((velo.shape[0], velo.shape[1], 3))
    angles = np.arange(0, 2* np.pi, 2 * np.pi / velo.shape[1])
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            x = velo[i][j] * np.cos(angles[j])
            y = velo[i][j] * np.sin(angles[j])
            out[i][j] = np.array([x, y, heights[i]])

    return out

def batch_pairwise_dist(a,b, gpu=0):
    x,y = a,b
    if x.size(-1) != 3 : x = x.transpose(2,1)
    if y.size(-1) != 3 : y = y.transpose(2,1)
    if x.size(-2) != y.size(-2) :
        minn, maxx = (x,y) if x.size(-2) < y.size(-2) else (y,x)
        indices = np.random.choice(int(maxx.size(-2)), int(minn.size(-2)))
        indices = torch.from_numpy(indices).cuda().long()
        maxx = maxx[:, indices]
        x, y = minn, maxx
    bs, num_points, points_dim = x.size()
    xx = torch.bmm(x, x.transpose(2,1))
    yy = torch.bmm(y, y.transpose(2,1))
    zz = torch.bmm(x, y.transpose(2,1))
    diag_ind = torch.arange(0, num_points).type(torch.LongTensor).cuda(gpu)
    rx = xx[:, diag_ind, diag_ind].unsqueeze(1).expand_as(xx)
    ry = yy[:, diag_ind, diag_ind].unsqueeze(1).expand_as(xx)#(yy)
    P = (rx.transpose(2,1) + ry - 2*zz)
    return P


def batch_NN_loss(x, y, dim=1):
    assert dim != 0
    dist = batch_pairwise_dist(x,y)
    values, indices = dist.min(dim=dim)
    return values.mean(dim=-1)


def Chamfer_Dist(x, y, gpus=None):
    gpu = 0
    if gpus is not None : 
        current_gpu, empty_gpu = gpus
        gpu = empty_gpu
        x = x.cuda(empty_gpu)
        y = y.cuda(empty_gpu)
    dist = batch_pairwise_dist(x,y, gpu=gpu)
    values_a, _ = dist.min(dim=1)
    values_b, _ = dist.min(dim=2)
    answer = values_a.mean(dim=-1) + values_b.mean(dim=-1)
    if gpus is not None :
        answer = answer.cuda(current_gpu)
    
    return answer


def wasserstein_dist(x, y, return_tensor=False):
    # pytorch tensor is a bs x N x N matrix
    distances = batch_pairwise_dist(x,y)
    bs, set_size, _ = distances.size()
    dist_norm = distances / distances.max()
    # we need to scale the distances to have mea
    np_dist = dist_norm.cpu().data.numpy()
    a, b = [ot.unif(x.size(1))] * 2
    indices = ot.batch_emd(a, b, np_dist)
    indices = torch.from_numpy(indices).cuda()
    max_val, max_ind = indices.max(-1)
    # max_ind has shape bs x N
    index_v = Variable(max_ind).unsqueeze(2)
    pair_distances = torch.gather(distances, dim=2, index=index_v)
    return pair_distances if return_tensor else pair_distances.mean()
    
    
def pairwise_dist(x, y):
    xx, yy, zz = torch.mm(x,x.t()), torch.mm(y,y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))
    P = (rx.t() + ry - 2*zz)
    return P


def NN_loss(x, y, dim=0):
    dist = pairwise_dist(x, y)
    values, indices = dist.min(dim=dim)
    return values.mean()





def ortho_penalty(transforms):
    running_penalty = 0
    for trans in transforms : 
        bs, _, K = trans.size()
        mat_diff = torch.bmm(trans, trans.transpose(2,1))
        mat_diff = mat_diff - Variable(torch.eye(K).cuda().unsqueeze(0).repeat(bs, 1, 1))
        running_penalty += mat_diff.mean()
    return running_penalty



# (x - y)^2 = x^2 - 2*x*y + y^2
def similarity_matrix(mat, y):
    # get the product x * y
    # here, y = x.t()
    r = torch.mm(mat, y.t()) # mat.t())
    # get the diagonal elements
    diag = r.diag().unsqueeze(0)
    diag = diag.expand_as(r)
    # compute the distance matrix
    D = diag + diag.t() - 2*r
    return D.sqrt()



def to_delta(tensor):
    # bs, seq_len, C, H, W = tensor.size()
    a = tensor[:, 1:]
    b = tensor[:, :-1]
    return a - b
 

def to_delta_extra(tensor):
    return tensor[:, 1:]


def from_delta(tensor, first_frame, og=None):
    bs, seq_len, C, H, W = tensor.size()
    new_tensor = -100*torch.ones(bs, seq_len+1, C, H, W).cuda()
    new_tensor[:, 0] = first_frame
    for i in range(1, seq_len+1):
        new_tensor[:, i] = new_tensor[:, i-1] + tensor[:, i-1]
    return new_tensor
        


# applies softmax on the channel dim of b_s, C, H, W tensor
def softmax_4D(tensor):
    bs, C, H, W = tensor.size()
    # put channels at the end
    tensor = tensor.permute(0,2,3,1)
    tensor = tensor.contiguous().view(bs*H*W, C)
    tensor = nn.Softmax()(tensor)
    tensor = tensor.view(bs, H, W, C)
    tensor = tensor.permute(0, 3, 1, 2)
    return tensor
   

# calculates cross entropy loss on channel dim of
# b_s, C, H, W tensor ans b_s, H, W
def CrossEntropy_4D(CE_criterion, tensor_a, tensor_b):
    bs, C, H, W = tensor_a.size()
    # put channels at the end
    tensor_a = tensor_a.permute(0,2,3,1)
    tensor_b = tensor_b.permute(0,2,3,1)
    tensor_a = tensor_a.contiguous().view(bs*H*W, C)
    tensor_b = tensor_b.contiguous().view(bs*H*W,)
    return CE_criterion(tensor_a, tensor_b)

# converts b_s, seq_len, H, W category numpy tensor 
# to b_s, seq_len, num_cat, H, W numpy tensor
def one_hot_4D(tensor, C=3):
    return (np.arange(C) == tensor[:, :, :, :, None]).astype('float32')

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def calc_gradient_penalty(netD, real_data, fake_data, batch_size=50, gpu=0):
    x, y = real_data, fake_data
    '''
    if x.size(-1) != 3 : x = x.transpose(2,1)
    if y.size(-1) != 3 : y = y.transpose(2,1)
    if x.size(-2) != y.size(-2) :
        minn, maxx= (x,y) if x.size(-2) < y.size(-2) else (y,x)
        #indices = np.random.choice(int(maxx.size(-2)), int(minn.size(-2)))
        #indices = torch.from_numpy(indices).cuda().long()
        maxx = maxx[:, :minn.size(-2)]#indices]
        real_data, fake_data = minn, maxx
    '''
    alpha = torch.rand(batch_size, 1, 1, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda(gpu)
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = interpolates.cuda(gpu)#.transpose(2,1)
    interpolates = autograd.Variable(interpolates, requires_grad=True)
    disc_interpolates = netD(interpolates)
    disc_interpolates = disc_interpolates[0] if len(disc_interpolates) > 1 else disc_interpolates
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates, 
                              grad_outputs = torch.ones(disc_interpolates.size()).cuda(gpu), 
                              create_graph = True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients + 1e-16
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()


def imshow(img, grid=False, epoch=0, display=True):
    npimg = img.numpy()
    npimg = np.transpose(npimg, (1,2,0))
    if npimg.shape[-1] == 3 or grid: 
        plt.imshow(npimg)
    else : 
        plt.imshow(npimg.reshape((npimg.shape[0], npimg.shape[1])))
    if display : 
        plt.draw()
        plt.pause(4)
    else : 
        plt.savefig('images/'+str(epoch)+'.png')


def show_seq(seq, epoch=0, display=True):
    imshow(torchvision.utils.make_grid(seq), epoch=epoch, grid=True, display=display)


def load_kitti(bbs=100, batch_size=32, seq_len=10, skip=1, big=2, use_extra=True, train='train'):
    import hickle as hkl
    velocity = [8,9,10]
    extra_indices = [19]
    X = hkl.load('../prednet/kitti_data/X_'+train+'_lidar.hkl')
    names = hkl.load('../prednet/kitti_data/sources_'+train+'_lidar.hkl')
    X_extra = hkl.load('../prednet/kitti_data/X_'+train+'_extra_lidar.hkl')
    C ,H, W = 1, 128, 176 
    while True :
        if use_extra : 
            batches = np.zeros((bbs * batch_size, seq_len, C, H, W))
            batches_extra = np.zeros((bbs * batch_size, seq_len, 2))
        else : 
            batches = np.zeros((bbs * batch_size, seq_len, C , H, W))

        for i in range(batch_size * bbs):
            found_valid_batch = False
            while not found_valid_batch: 
                index = np.random.randint(0, X.shape[0] - skip * seq_len)
                indices = slice(index, index  + skip * seq_len)
                # make sure they are all form the same video
                batch = X[indices]
                if use_extra : 
                    extra = X_extra[indices]
                # if batch.shape[0] != skip * seq_len : continue
                name = names[indices]
                for j in range(batch.shape[0]-1):
                    if name[j] != name[j+1] : 
                        break # start inner while loop again
                found_valid_batch = True
                
            if use_extra :
                shp = batch.shape
                if batches[i].shape[0]*2 != batch.shape[0] : pdb.set_trace()
                batches[i] = batch.reshape((shp[0], 1, shp[1], shp[2]))[::skip]
                speeds = extra[:, velocity][::skip]
                speed = np.sqrt(speeds[:,0]**2 + speeds[:,1]**2 + speeds[:,2]**2)
                batches_extra[i] = extra[:, extra_indices][::skip]
                batches_extra[i, :, 0] = speed
            else :
                batches[i] = batch.transpose(0,3,1,2)[::skip]
        
        if use_extra : 
            batches_extra = batches_extra
            shp = batches_extra.shape
            # batches_extra = ([batches_extra[:, :, i].reshape((shp[0], shp[1], 1))
            #                  for i in range(shp[2])])

        if True  : 
            # scale it 
            batches_extra[:, :, 0] /= 10.
            batches_extra[:, :, 1] *= 15.
        yield batches.astype('float32') / 255. , batches_extra


def load_kitti_lidar(bbs=100, batch_size=32, train='train'):
    import hickle as hkl
    # X = hkl.load('../prednet/kitti_data/X_'+train+'_lidar_raw.hkl')
    X = hkl.load('data.hkl')
    X = X * 2 # [-.5, .5] --> [-1, 1]
    datap, n, c = X.shape
    # scale each 3 dimension so that they E (0, 1)
    while True :
        batches = np.zeros((bbs * batch_size, n, c))
        indices = np.random.choice(X.shape[0], bbs * batch_size)
        batches = X[indices]
        # batches[:, :, 0] /= 80.
        # batches[:, :, 1] /= 80.
        # batches[:, :, 2] /= 30.
        yield batches.astype('float32')


def draw_point_cloud(velo_frame, ax, title, axes=[0, 1, 2], xlim3d=None, ylim3d=None, zlim3d=None):
    ax.scatter(*np.transpose(velo_frame[:, axes]), s=point_size, cmap='gray')
    if len(axes) > 2:
        ax.set_ylim3d(*axes_limits[axes[1]])
        ax.set_zlim3d(*axes_limits[axes[2]])
        ax.set_zlabel('{} axis'.format(axes_str[axes[2]]))
    else : 
        ax.set_xlim(*axes_limits[axes[0]])
        ax.set_ylim(*axes_limits[axes[1]])


def post_process(velo):
    velo[:, 0] *= 40 #80.
    velo[:, 1] *= 40 #80. 
    velo[:, 2] *= 30.
    return velo


def lidar_to_img(velo, epoch=0):
    # velo = post_process(velo)
    velo *= 20 
    f, ax3 = plt.subplots(1,1, figsize=(5,5))
    draw_point_cloud(velo, ax3, '', axes=[0,1])
    plt.axis('off')
    # plt.show()
    plt.savefig('images/' + str(epoch) + '.png', bbox_inches='tight')
    plt.close()


def iter_minibatches(inputs, batch_size, extra=None, forever=False):
    while True : 
        for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
            excerpt = slice(start_idx, start_idx + batch_size)
            if extra is not None : 
                yield (inputs[excerpt], extra[excerpt]) # [x[excerpt] for x in extra])
            else : 
                yield inputs[excerpt]
        if not forever : 
            break
   
