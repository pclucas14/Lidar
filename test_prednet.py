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
from models_2 import * 

load_weights = False#True
seq_len = 6
batch_size = 16
future = 2
C, H, W = 1, 128, 176 #128, 160
big = 1
bbs = 100 
extra = [] # [1, 1, 12, 1]
input_shape = (batch_size, C, H, W) 

A_channels = (C, 48, 96, 192)
R_channels = A_channels
A_filt_sizes = (3, 3, 3)
Ahat_filt_sizes = (3, 3, 3, 3)
R_filt_sizes = (3, 3, 3, 3)
clip = 100

target_index = seq_len if future == 0 else slice(seq_len,seq_len+future+1)
test_index = slice(seq_len-3,seq_len+1) if future == 0 else slice(seq_len+future-3, seq_len+future+1)
tensor_size = 1 if future == 0 else future + 1
model = PredNet(input_shape, A_channels, R_channels, A_filt_sizes, 
                Ahat_filt_sizes, R_filt_sizes, extra=extra)
model.cuda()
if load_weights : 
    model.load_state_dict(torch.load('models/pn_skip1_epoch_1200.pth'))
    print 'PredNet weights loaded'
else : 
    model.apply(weights_init)

'''
test for generating data  (SRGAN)
'''
# build_srgan_dataset(model, size=50000)

generator_train = load_kitti(bbs=bbs, batch_size=batch_size, skip=2, seq_len=seq_len+1+future) # 2 in other one
# generator_train = load_car_data(bbs=bbs, batch_size=batch_size, skip=10, extra=extra, seq_len=seq_len+1+future)
opt = optim.Adam(model.parameters(), lr=1e-4)
input = torch.FloatTensor(batch_size, seq_len, C, H, W).cuda()
target = torch.FloatTensor(batch_size, tensor_size, C, H, W).cuda()
extras = [torch.FloatTensor(batch_size, seq_len+1+future, extra[i]).cuda() for i in range(len(extra))]
criterion = nn.MSELoss()
for epoch in range(5000):
    print epoch
    r_loss, r_loss_inv = 0., 0.
    data, extra = next(generator_train)
    # data = next(generator_train)
    # mb_iterator = iter_minibatches(data, batch_size, forever=False)
    for i in range(bbs):
        
        ''' train with matching data '''
        index = slice(i * batch_size, (i+1) * batch_size)
        data_b, extra_b = data[index], [ex_i[index] for ex_i in extra]
        input.copy_(torch.from_numpy(data_b[:, :seq_len, :, :, :]))
        target.copy_(torch.from_numpy(data_b[:, target_index, :, :, :]))

        for i in range(len(extras)):
            extras[i].copy_(torch.from_numpy(extra_b[i]))

        input_v = Variable(input)
        target_v = Variable(target)
        extra_v = [Variable(extras[i]) for i in range(len(extras))]
        opt.zero_grad()
        out = model(input_v, future=future, return_errors=0, extra=extra_v)
        loss = criterion(out, target_v)
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), clip)
        opt.step()
        r_loss += loss.data[0]

        ''' train with random data '''
        index = np.random.random_integers(data.shape[0], shape=(batch_size,))
        data_b, extra_b = data[index], [ex_i[index] for ex_i in extra]
        input.copy_(torch.from_numpy(data_b[:, :seq_len, :, :, :]))
        target.copy_(torch.from_numpy(data_b[:, target_index, :, :, :]))

        for i in range(len(extras)):
            extras[i].copy_(torch.from_numpy(extra_b[i]))

        input_v = Variable(input)
        target_v = Variable(target)
        extra_v = [Variable(extras[i]) for i in range(len(extras))]
        opt.zero_grad()
        out_fake = model(input_v, future=future, return_errors=0, extra=extra_v)
        loss = criterion(out_fake, target_v)
        loss = loss * -.1 
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), clip)
        opt.step()
        r_loss_inv += loss.data[0]

        
    print("%.2f" % r_loss)
    print("%.2f" % r_loss_inv)
    print ""
    input_v = input_v.squeeze(2)
    target_v = target_v.squeeze(2)
    if future > 0 : out = out.squeeze()
    try_something = torch.cat([target_v, out, out_fake], dim=1)[:10]
    out = [try_something[i] for i in range(try_something.size(0))]
    out = torch.cat(out, dim=1)
    out = [out[i] for i in range(out.size(0))]
    out = torch.cat(out, dim=1)
    out = out.cpu().data.numpy() * 255.
    out = out.astype('uint8')
    Image.fromarray(out).save('images/' + str(epoch) + '.png')

    if epoch % 50 == 0 : 
        torch.save(model.state_dict(), '%s/pn_lidar%d.pth' % ('models', epoch))

pdb.set_trace()

# skip1 and skip2 has seq len = 8 future=0
# future2(3) has skip1 ans seqlen 7
