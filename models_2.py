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

indices = None

class nnetG(nn.Module):
    def __init__(self, ngpu, nz=100, ngf=64, nc=3, base=4, ff=(3,16)):
        super(nnetG, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(     nz, ngf * 8, ff, 1, 0, bias=False), # 3 was a 4
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),
                # state size. (ngf*8) x 4 x 4
                nn.ConvTranspose2d(ngf * 8, ngf * 4, (3,4), stride=2, padding=(0,1), bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
                # state size. (ngf*4) x 8 x 8
                nn.ConvTranspose2d(ngf * 4, ngf * 2, (3,4), 2, padding=(0,1), bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
                # state size. (ngf*2) x 16 x 16
                nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),
                # state size. (ngf) x 32 x 32
                nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
                #nn.Tanh()
                nn.Sigmoid()
                # state size. (nc) x 64 x 64
                )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            input = input.unsqueeze(2).unsqueeze(3)
            output = self.main(input)
        return output



class nnetD(nn.Module):
    def __init__(self, ngpu, ndf=64, nc=3, nz=1, lf=(3,16)):
        super(nnetD, self).__init__()
        self.encoder = True if nz > 1 else False
        self.ngpu = ngpu
        self.main = nn.Sequential(
                # input is (nc) x 64 x 64
                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf) x 32 x 32
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*2) x 16 x 16
                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*4) x 8 x 8
                nn.Conv2d(ndf * 4, ndf * 8, (3,4), 2, (0,1), bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*8) x 4 x 4
                )
        self.main_ = nn.Sequential(
                nn.Conv2d(ndf * 8, nz, lf, 1, 0, bias=False))

    def forward(self, input, return_hidden=False):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            if input.size(-1) == 3: input = input.transpose(1, 3)
            output_tmp = self.main(input)
            output = self.main_(output_tmp)
        if return_hidden : 
            return output, output_tmp
        return output if self.encoder else output.view(-1, 1).squeeze(1) 



class T_Net(nn.Module):
    def __init__(self, num_points, K_in=3, K_out=3):
        super(T_Net, self).__init__()
        self.num_points = num_points
        self.K_in, self.K_out = K_in, K_out
        self.conv1 = torch.nn.Conv1d(K_in, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.mp1 = torch.nn.MaxPool1d(num_points)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, K_out*K_out)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.mp1(x)
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.K_out).flatten().astype(np.float32)))
        iden = iden.view(1, self.K_out*self.K_out).repeat(batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.K_out, self.K_out)
        return x



class PointNet(nn.Module):
    def __init__(self, num_points = 2500, global_feat = True):
        super(PointNet, self).__init__()
        self.input_transform = T_Net(num_points, K_in=3, K_out=3)
        self.feat_transform = T_Net(num_points, K_in=64, K_out=64)

        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        
        self.conv3 = torch.nn.Conv1d(64, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, 256, 1)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)

        self.conv5 = torch.nn.Conv1d(256+3, 64, 1)
        self.conv6 = torch.nn.Conv1d(64, 1, 1)
        self.bn5 = nn.BatchNorm1d(64)
        self.bn6 = nn.BatchNorm1d(1)

        self.fc1 = nn.Linear(num_points, 512)
        self.fc2 = nn.Linear(512, 1)

        self.mp1 = torch.nn.MaxPool1d(num_points)
        self.drop = nn.Dropout()
        self.num_points = num_points

    def forward(self, input):
        # input has shape bs x N x 3
        input = input.transpose(2,1)
        # input has shape bs x 3 X N
        x = input
        batchsize = x.size()[0]
        input_trans = self.input_transform(x)
        # apply input transform
        x = x.transpose(2,1)
        x = torch.bmm(x, input_trans)
        x = x.transpose(2,1)
        # project to x64
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        feat_trans = self.feat_transform(x)
        # apply feat transform
        x = x.transpose(2,1)
        x = torch.bmm(x, feat_trans)
        x = x.transpose(2,1)
        # bs x N x 64
        # project to x256
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        # extract global features
        global_feat = self.mp1(x)
        # repeat and concatenate to input
        global_feat = global_feat.view(-1, 256, 1).repeat(1, 1, self.num_points)
        # bs x 512 x N
        x = torch.cat([input, global_feat], 1)
        # bs x 515 x N
        # reduce dim 
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = x.squeeze()
        # bs x N
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)

        transformations = [input_trans, feat_trans] 
        return x, transformations


class STN3d(nn.Module):
    def __init__(self, num_points = 2500):
        super(STN3d, self).__init__()
        self.num_points = num_points
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.mp1 = torch.nn.MaxPool1d(num_points)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.mp1(x)
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class PointNetfeat(nn.Module):
    def __init__(self, num_points = 2500, global_dim=512):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d(num_points = num_points)
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, global_dim, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(global_dim)
        self.mp1 = torch.nn.MaxPool1d(num_points)
        self.num_points = num_points
        self.global_dim = global_dim
    
    
    def forward(self, x):
        batchsize = x.size()[0]
        trans = self.stn(x)
        x = x.transpose(2,1)
        x = torch.bmm(x, trans)
        x = x.transpose(2,1)
        x = F.relu(self.bn1(self.conv1(x)))
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = self.mp1(x)
        x = x.view(-1, self.global_dim)
        return x # , trans


class PointNetDisc(nn.Module):
    def __init__(self, num_points = 2500, k = 2):
        super(PointNetDisc, self).__init__()
        self.num_points = num_points
        self.feat = PointNetfeat(num_points, global_dim=512)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, k)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
    def forward(self, x):
        # my line
        x = x.transpose(2,1)
        x = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x


''' permutation invariant layer '''
class PI_Layer(nn.Module):

    def __init__(self, input_chan, output_chan, num_points, nonlinearity=nn.Tanh()):
        super(PI_Layer, self).__init__()

        self.nonlinearity = nonlinearity
        self.num_points = num_points
        self.conv = nn.Conv1d(input_chan, output_chan, 1)
        self.mp = nn.MaxPool1d(num_points)


    def forward(self, input):
       # input should have size bs x dim x N (e.g. 32 x 3 x 2500)
       max_val = self.mp(input)
       val = input - max_val
       val = self.conv(val)
       val = self.nonlinearity(val)
       return val


''' feature encoder / disc '''
class PC_Disc(nn.Module):
    def __init__(self, input_shape, channels=[256, 512, 1024], discriminate=False):
        super(PC_Disc, self).__init__()

        bs, num_pts, dim = input_shape
        channels.insert(0, dim)
        layers = [PI_Layer(channels[i], channels[i+1], num_pts, nonlinearity=nn.ReLU()) 
                  for i in range(len(channels)-1)]
        self.mp = nn.MaxPool1d(num_pts)
        self.main = nn.Sequential(*layers)
        self.discriminate = discriminate

        if discriminate : 
            self.dense = nn.Linear(channels[-1], 1)

    def feat_extract(self, input):
        if input.size(2) == 3 : input = input.transpose(2,1)
        val = self.main(input)
        val = self.mp(val).squeeze()
        return val 

    def forward(self, input):
        val = self.feat_extract(input)
        if self.discriminate: 
            return self.dense(val)
        else : 
            return val
        

'''
class CLSTM_cell(nn.Module):

    def __init__(self, input_shape, output_shape, filter_size, extra=[]):
	super(CLSTM_cell, self).__init__()

	self.input_shape = input_shape
	self.output_shape = output_shape
	padding = (filter_size - 1) / 2
	self.conv = nn.Conv2d(input_shape[1] + output_shape[1] + len(extra), 4 * output_shape[1], filter_size, 1, padding)
        if len(extra) > 0 :
            # we need a convolution that project [speed, angle] --> hidden_channel ** 2
            dense_extra = []
            for i in range(len(extra)):
                dense_extra.append(nn.Linear(extra[i], output_shape[-2] * output_shape[-1]))
            self.dense_extra = nn.ModuleList(dense_extra)


    # extra is FloatTensor [steering angle, speed] (already normalized)
    def forward(self, input, hidden_state, extra=[]):
	h_t, c_t = hidden_state
        if len(extra) > 0 :  
            squares = []
            for i in range(len(extra)):
                square_i = self.dense_extra[i](extra[i])
                square_i = nn.ReLU()(square_i)
                squares.append(square_i.view(-1, self.output_shape[-2], self.output_shape[-1]))
            squares = torch.stack(squares, 1)

            try : 
                combined = torch.cat([input, h_t, squares], 1)
            except : 
                pdb.set_trace()
        else : 
	    combined = torch.cat((input, h_t), 1)
        try : 
            all_conv = self.conv(combined)
        except : 
            pdb.set_trace()

	i, f, o, c_tild = torch.split(all_conv, self.output_shape[1], dim=1)
	
	i = torch.sigmoid(i)
	f = torch.sigmoid(f)
	o = torch.sigmoid(o)
	c_tild = torch.tanh(c_tild)

	next_c = f * c_t + i * c_tild
	next_h = o * torch.tanh(next_c)
	return next_h, next_c


    def init_hidden(self):
	a, b, c, d = self.output_shape
	return (Variable(torch.zeros(a, b, c, d)).cuda(), 
		Variable(torch.zeros(a, b, c, d)).cuda())
'''     
class CLSTM_cell(nn.Module):

    def __init__(self, input_shape, output_shape, filter_size, stride=1, extra_dim=0, gpu=0):
	super(CLSTM_cell, self).__init__()
        if stride == 1 : 
            assert filter_size % 2 == 1, 'stride 1 conv. requires odd filter size'
            padding = (filter_size - 1) / 2
        elif stride == 2 : 
            assert filter_size % 2 == 0, 'stride 2 conv. requires even filter size'
	    padding = filter_size / 2 - 1
        else : 
            raise NotImplementedError
        
        if extra_dim > 0 : 
            # if additional info is used (e.g. speed, steering angle), 
            # we double the channels of the input
            extra_in = input_shape[1]
            # we then add the dense layer to map from extra info --> channels
            self.dense_extra = nn.Linear(extra_dim, extra_in)
            # 1x1 conv to blend state with actions
            self.blend = nn.Conv2d(input_shape[1] + extra_in, 
                                   input_shape[1] + extra_in, 
                                   1, 
                                   stride=1, 
                                   padding=0)
        else : 
            extra_in = 0 

        self.input_shape = input_shape
	self.output_shape = output_shape
        self.extra_dim = extra_dim
        self.gpu = gpu

        if input_shape[-1] < output_shape[-1] : # decoder
            self.conv_input = nn.ConvTranspose2d(input_shape[1] + extra_in, 
                                                 4 * output_shape[1], 
                                                 filter_size,
                                                 stride=stride, 
                                                 padding = padding)
            #                           h_t.size()
 	    self.conv_hidden = nn.Conv2d(output_shape[1], 
                                    4 * output_shape[1], 
                                    filter_size + 1, 
                                    stride=1, 
                                    padding=(filter_size / 2))#filter_size/2)
        elif input_shape[-1] == output_shape[-1] : 
            self.conv_input = nn.Conv2d(input_shape[1] + extra_in, 
                                        4 * output_shape[1], 
                                        filter_size, 
                                        stride=1, 
                                        padding=(filter_size-1) / 2)
            self.conv_hidden = nn.Conv2d(output_shape[1], 
                                         4 * output_shape[1], 
                                         filter_size, 
                                         stride=1, 
                                         padding=(filter_size-1) / 2)
        else : 
            #                           x_t.size(1)      f(extra).size(1)
            self.conv_input = nn.Conv2d(input_shape[1] + extra_in, 
                                        4 * output_shape[1], 
                                        filter_size, 
                                        stride=stride, 
                                        padding=padding)
            #                           h_t.size()
            self.conv_hidden = nn.Conv2d(output_shape[1], 
                                         4 * output_shape[1], 
                                         filter_size + 1, 
                                         stride=1, 
                                         padding=(filter_size / 2))#filter_size/2)
             
       

    def forward(self, input, hidden_state, extra=None):
	h_t, c_t = hidden_state
        if self.extra_dim > 0 : 
            assert self.extra_dim == extra.size(1)
            f_extra = self.dense_extra(extra)
            bs, dim = f_extra.size()
            f_extra = f_extra.view(bs, dim, 1, 1)
            f_extra = f_extra.expand(bs, dim, self.input_shape[-2], self.input_shape[-1])
            try : 
                combined_input = torch.cat([input, f_extra], 1)
                input = self.blend(combined_input)
            except : 
                pdb.set_trace()
        try :  
            input_conved = self.conv_input(input)
            hidden_conved = self.conv_hidden(h_t)
        except : 
            pdb.set_trace()
        
        input_conved = torch.split(input_conved, self.output_shape[1], dim=1)
        hidden_conved = torch.split(hidden_conved, self.output_shape[1], dim=1)
        try : 
            i, f, o, c_tild = [x + y for (x,y) in zip(input_conved, hidden_conved)]
        except : 
            pdb.set_trace()
	i = torch.sigmoid(i)
	f = torch.sigmoid(f)
	o = torch.sigmoid(o)
	c_tild = torch.tanh(c_tild)

	next_c = f * c_t + i * c_tild
	next_h = o * torch.tanh(next_c)
	return next_h, next_c


    # TODO : add multi-gpu support through constructor
    def init_hidden(self):
	a, b, c, d = self.output_shape
	return (Variable(torch.zeros(a, b, c, d)).cuda(self.gpu), 
		Variable(torch.zeros(a, b, c, d)).cuda(self.gpu))
    

class Highway_Layer(nn.Module):

    def __init__(self, shape, filter_size=5):
        super(Highway_Layer, self).__init__()
        # note : not exaclty highway layer, more of a weighted/conditional
        # mixture of 2 inputs
        bd, C, H, W = shape
        padding = (filter_size - 1) / 2
        self.gate_conv = nn.Conv2d(C*2, C, 5, padding=padding)


    def forward(self, a, b):
        concat = torch.cat([a,b], dim=1)
        gate = self.gate_conv(concat)
        gate = torch.sigmoid(gate)
        return a * gate + b * (1 - gate)


class Highway_Gen(nn.Module):

    def __init__(self, input_shape, channels=[64, 128], gpu=0, extra_dim=0):
        super(Highway_Gen, self).__init__()

        bs, seq_len, C, H, W = input_shape
        enc, dec, gates = [], [], []
        channels.insert(0, input_shape[2])

        for i in range(len(channels)-1):
            input_dim =  (bs, channels[i]  , H // (2**i)    , W // (2**i))
            output_dim = (bs, channels[i+1], H // (2**(i+1)), W // (2**(i+1)))
            enc.append(CLSTM_cell(input_dim,
                                  output_dim,
                                  4,
                                  stride=2,
                                  extra_dim=extra_dim, 
                                  gpu=gpu))
            dec.insert(0, CLSTM_cell(
                                  output_dim, 
                                  input_dim, 
                                  4,
                                  stride=2,
                                  extra_dim=extra_dim, 
                                  gpu=gpu))
            # we have 1 gate for every dimension, including start dim
            gates.insert(0, Highway_Layer(input_dim))
        
        # add the remaining gate (for bottleneck layer)
        # edit : not needed. gates.insert(0, Highway_Layer(output_dim))
        
        modules = [enc, dec, gates]
        self.enc, self.dec, self.highway = [nn.ModuleList(mod) for mod in modules]


    def step(self, input, hid_t, extra=None):
        intermediate_outputs = []
        val = input

        for i in range(len(self.enc)):
            cell = self.enc[i]
            h_i, c_i = cell(val, hid_t[i], extra=extra)
            # update the hidden states at layer i
            hid_t[i] = (h_i, c_i)
            # update the input for next layer
            val = h_i
            # keep deeper values in front
            if i != len(self.enc) -1 : intermediate_outputs.insert(0, val)

        for i in range(len(self.dec)):
            cell = self.dec[i]
            h_i, c_i = cell(val, hid_t[i+len(self.enc)], extra=extra)
            # update the hidden states at layer i 
            hid_t[i+len(self.enc)] = (h_i, c_i)
            # update the input for next layer
            val = h_i
            if i < len(self.dec)-1: 
                gate = self.highway[i]
                val = gate(val, intermediate_outputs[i])

        # last = self.highway[-1](torch.sigmoid(val), input)
        # return torch.sigmoid(last), hid_t
        return torch.sigmoid(val), hid_t


    def forward(self, input, hid=None, extra=None, future=0):
        if hid is None : 
            hid =  [cell.init_hidden() for cell in self.enc]
            hid += [cell.init_hidden() for cell in self.dec]

        for i in range(input.size(1)):
            x_t = input[:, i]
            extra_t = extra[:, i] if extra is not None else None
            out, hid = self.step(x_t, hid, extra=extra_t)

        if future == 0 : 
            return out.unsqueeze(dim=1)
        else : 
            preds = [out]
            for i in range(future):
                extra_t = extra[:, i+input.size(1)] if extra is not None else None
                out, hid = self.step(out, hid, extra=extra_t)
                preds.append(out)

            return torch.stack(preds, dim=1)


class RGEN(nn.Module):

    def __init__(self, input_shape, channels=[32, 64, 128], lstm_layers=2, gpu=0, extra_dim=[]):
        super(RGEN, self).__init__()

        self.input_shape = input_shape
        bs, seq_len, C, H, W = input_shape
        ds_factor = 2 ** len(channels)
        bottleneck = (bs, channels[-1], H // ds_factor, W // ds_factor)
        # channels.insert(0, C)

        self.enc = Encoder((bs, C, H, W), channels)
        self.dec = Decoder((bs, C, H, W), channels[::-1])
        self.clstms = nn.ModuleList(
                [CLSTM_cell(bottleneck, bottleneck, 5, extra_dim=extra_dim) 
                    for _ in range(lstm_layers)])


    def step(self, x_t, hid_t, decode=False, extra=None):
        # first we downsize input through encoder
        downsized = self.enc(x_t)
        input = downsized

        # next we feed it through lstms
        for i in range(len(self.clstms)):
            h_i, c_i = self.clstms[i](input, hid_t[i], extra=extra)
            # update the hidden states at layer i 
            hid_t[i] = (h_i, c_i)
            # update the input for the next layer
            input = h_i 

        if decode : 
            out = self.dec(h_i)
            return hid_t, out
        else : 
            return hid_t


    def forward(self, input, future=0, hid_t=None, extra=None):
        if hid_t is None : 
            hid_t = [self.clstms[i].init_hidden() for i in range(len(self.clstms))]        

        for i in range(input.size(1)-1):
            x_t = input[:, i]
            extra_t = extra[:, i] if extra is not None else None
            # extra_t = [ex[:, i, :] for ex in extra] if extra is not None else None 
            hid_t = self.step(x_t, hid_t, decode=False, extra=extra_t)

        extra_t = extra[:, input.size(1)-1] if extra is not None else None
        # extra_t =[ex[:, input.size(1)-1, :] for ex in extra] if extra is not None else None
        hid_t, pred = self.step(input[:, -1], hid_t, decode=True, extra=extra_t)

        if future == 0 : 
            return pred.unsqueeze(dim=1)
        else : 
            preds = [pred]

        for i in range(future):
            extra_t = extra[:, i+input.size(1)] if extra is not None else None
            # extra_t = [ex[:, i+input.size(1), :] for ex in extra] if extra is not None else None
            hid_t, pred = self.step(pred, hid_t, decode=True, extra=extra_t)
            preds.append(pred)

        return torch.stack(preds, dim=1)



class RDISC(nn.Module):

    def __init__(self, input_shape, channels=[32, 64, 128, 256], lstm_layers=[1,1], gpu=0, extra_dim=[]):
        super(RDISC, self).__init__()

        self.input_shape = input_shape
        bs, seq_len, C, H, W = input_shape
        ds_factor = 2 ** len(channels)
        bottleneck = (bs, channels[-1], H // ds_factor, W // ds_factor)

        self.enc = Encoder((bs, C, H, W), channels)
        self.clstms_input = nn.ModuleList(
                [CLSTM_cell(bottleneck, bottleneck, 5, extra=extra_dim) 
                    for _ in range(lstm_layers[0])])
        self.clstms_pred  = nn.ModuleList(
                [CLSTM_cell(bottleneck, bottleneck, 5, extra=extra_dim) 
                    for _ in range(lstm_layers[1])])
        self.dense = nn.Linear(np.prod(bottleneck[1:]), 1)


    def forward(self, input, pred, hid_=None, extra=None):
        '''
        TODO : make sure the right timestep is used when feeding extras. 
        check prednet to see how its done. 
        '''
        if hid_ is None : 
            hid_ = [self.clstms_input[i].init_hidden() for i in range(len(self.clstms_input))]

            # 1) iterate over timesteps of input
            for i in range(input.size(1)):
                x_t = input[:, i]
                extra_t = extra[:, i] if extra is not None else None
                input_rec = self.enc(x_t)

                # iterate over layers  
                for j in range(len(self.clstms_input)):
                    cell = self.clstms_input[j]
                    h_j, c_j = cell(input_rec, hid_[j], extra=extra_t)
                    # update hidden_states
                    hid_[j] = (h_j, c_j)
                    input_rec = h_j
            
        # if hidden states are given, we assume first part has already been done
        hid_first_part = [h for h in hid_]
        # 2) iterate over the real/fake data (end of input)
        for i in range(pred.size(1)):
            pred_t = pred[:, i]
            input_rec = self.enc(pred_t)
            extra_t = extra[:, i+input.size(1)] if extra is not None else None
            # we simply transfer hidden states to the new cells
            # Note : this implies equal amt of lstm cells
            for j in range(len(self.clstms_pred)):
                cell = self.clstms_pred[j]
                h_j, c_j = cell(input_rec, hid_[j], extra=extra_t)
                # update hidden states
                hid_[j] = (h_j, c_j)
                input_rec = h_j

        # 3) push hidden state through dense layer to get final verdict
        return hid_first_part, self.dense(c_j.view(self.input_shape[0], -1))
# simple encoder structure to reduce input dimensionality



class Encoder(nn.Module):
    # channels : list of filters to use for each convolution (increasing order)
    # every layer uses stride 2 and divides dim / 2. 
    def __init__(self, input_shape, channels, filter_size=4, activation=nn.ReLU(), bn=False, last_nl=nn.ReLU()):
        super(Encoder, self).__init__()
        assert filter_size % 2 == 0
        self.input_shape = input_shape
        self.bn = bn
        padding = filter_size / 2 - 1
        self.convs = [nn.Conv2d(input_shape[1], channels[0], filter_size, stride=2, padding=padding)]
        for i in range(1, len(channels)):
            self.convs.append(nn.Conv2d(channels[i-1], channels[i], filter_size, stride=2, padding=padding))
        
        self.convs = nn.ModuleList(self.convs)
        self.activation = activation
        self.last_nl = last_nl

        if bn : 
            bns = []
            # no batch norm for first layer
            for i in range(1, len(channels)):
                bns.append(nn.BatchNorm2d(channels[i]))
            self.bns = nn.ModuleList(bns)

    def forward(self, input):
        sh = input.size()
        shrink = 2 ** len(self.convs)
        val = input.contiguous().view((sh[0]*sh[1], sh[2], sh[3], sh[4])) if len(sh) == 5 else input
        for i in range(len(self.convs)):
             val = self.convs[i](val)
             if self.bn and i != 0: # no batch norm for first layer 
                 val = self.bns[i-1](val)
             if i != len(self.convs) -1 : # if not last
                 val = self.activation(val)
             else : 
                 val = self.last_nl(val) if self.last_nl is not None else val
        
        val = val.view(sh[0], sh[1], -1, sh[3] / shrink, sh[4] / shrink) if len(sh) == 5 else val
        return val
        


# simple decoder structure to project back to original dimensions
class Decoder(nn.Module):
    # channels : list of filters to use for each convolution (DECreasing order)
    # every layer uses stride 2 and divides dim / 2. 
    def __init__(self, output_shape, channels, filter_size=4, activation=nn.ReLU(), bn=False, last_nl=nn.Sigmoid()):
        super(Decoder, self).__init__()
        assert filter_size % 2 == 0
        self.output_shape = output_shape
        self.bn = bn
        padding = filter_size / 2 - 1
        self.deconvs = []
        for i in range(len(channels)-1):
            self.deconvs.append(nn.ConvTranspose2d(channels[i], channels[i+1], filter_size, stride=2, padding=padding))
        
        self.deconvs.append(nn.ConvTranspose2d(channels[-1], output_shape[1], filter_size, stride=2, padding=padding))
        self.deconvs = nn.ModuleList(self.deconvs)
        self.activation = activation
        self.last_nl = last_nl

        if bn : 
            bns = []
            for i in range(1,len(channels)):
                # no batch norm for last layer
                bns.append(nn.BatchNorm2d(channels[i]))
            self.bns = nn.ModuleList(bns)


    def forward(self, input):
        sh = input.size()
        shrink = 2 ** len(self.deconvs)
        val = input.contiguous().view((sh[0]*sh[1], sh[2], sh[3], sh[4])) if len(sh) == 5 else input
        for i in range(len(self.deconvs)):
             val = self.deconvs[i](val)
             if i == len(self.deconvs)-1:
                 val = self.last_nl(val)
             else :
                 if self.bn :  
                     val = self.bns[i](val)
                 val = self.activation(val)

        val = val.view(sh[0], sh[1], -1, sh[3] * shrink, sh[4] * shrink) if len(sh) == 5 else val
        return val

