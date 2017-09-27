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


 
class CLSTM_cell(nn.Module):

    def __init__(self, input_shape, output_shape, filter_size, extra=[], gpu=0):
	super(CLSTM_cell, self).__init__()

	self.input_shape = input_shape
	self.output_shape = output_shape
        self.gpu = gpu
	padding = (filter_size - 1) / 2
	self.conv = nn.Conv2d(input_shape[1] + output_shape[1] + len(extra), 4 * output_shape[1], filter_size, 1, padding)
        if len(extra) > 0 :
            # we need a convolution that project [speed, angle] --> hidden_channel ** 2
            dense_extra = []
            for i in range(len(extra)):
                dense_extra.append(nn.Linear(extra[i], output_shape[-2] * output_shape[-1]))
            self.dense_extra = nn.ModuleList(dense_extra)


    def forward(self, input, hidden_state, extra=[]):
	h_t, c_t = hidden_state
        if len(extra) > 0 :  
            squares = []
            for i in range(len(extra)):
                try : 
                    square_i = self.dense_extra[i](extra[i])
                except : 
                    pdb.set_trace()
                square_i = nn.ReLU()(square_i)
                squares.append(square_i.view(-1, self.output_shape[-2], self.output_shape[-1]))
            squares = torch.stack(squares, 1)
            combined = torch.cat([input, h_t, squares], 1)
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


    # TODO : add multi-gpu support through constructor
    def init_hidden(self):
	a, b, c, d = self.output_shape
	return (Variable(torch.zeros(a, b, c, d)).cuda(self.gpu), 
		Variable(torch.zeros(a, b, c, d)).cuda(self.gpu))
    

class RGEN(nn.Module):

    def __init__(self, input_shape, channels=[32, 64, 128], lstm_layers=2, gpu=0, extra=[]):
        super(RGEN, self).__init__()

        self.input_shape = input_shape
        bs, seq_len, C, H, W = input_shape
        ds_factor = 2 ** len(channels)
        bottleneck = (bs, channels[-1], H // ds_factor, W // ds_factor)
        # channels.insert(0, C)

        self.enc = Encoder((bs, C, H, W), channels)
        self.dec = Decoder((bs, C, H, W), channels[::-1])
        self.clstms = nn.ModuleList(
                [CLSTM_cell(bottleneck, bottleneck, 5, extra=extra, gpu=gpu) 
                    for _ in range(lstm_layers)])


    def step(self, x_t, hid_t, decode=False, extra=[]):
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


    def forward(self, input, future=0, hid_t=None, extra=[]):
        if hid_t is None : 
            hid_t = [self.clstms[i].init_hidden() for i in range(len(self.clstms))]        

        for i in range(input.size(1)-1):
            x_t = input[:, i]
            extra_t = [ex[:, i, :] for ex in extra] if len(extra) > 0 else []
            hid_t = self.step(x_t, hid_t, decode=False, extra=extra_t)

        extra_t = [ex[:, input.size(1)-1, :] for ex in extra] if len(extra) > 0 else []
        hid_t, pred = self.step(input[:, -1], hid_t, decode=True, extra=extra_t)

        if future == 0 : 
            return pred
        else : 
            preds = [pred]

        for i in range(future):
            extra_t = [ex[:, i+input.size(1), :] for ex in extra] if len(extra) > 0 else []
            hid_t, pred = self.step(pred, hid_t, decode=True, extra=extra_t)
            preds.append(pred)

        return torch.stack(preds, dim=1)



class RDISC(nn.Module):

    def __init__(self, input_shape, channels=[32, 64, 128, 256], lstm_layers=[1,1], gpu=0, extra=[]):
        super(RDISC, self).__init__()

        self.input_shape = input_shape
        bs, seq_len, C, H, W = input_shape
        ds_factor = 2 ** len(channels)
        bottleneck = (bs, channels[-1], H // ds_factor, W // ds_factor)

        self.enc = Encoder((bs, C, H, W), channels)
        self.clstms_input = nn.ModuleList(
                [CLSTM_cell(bottleneck, bottleneck, 5, extra=extra, gpu=gpu) 
                    for _ in range(lstm_layers[0])])
        self.clstms_pred  = nn.ModuleList(
                [CLSTM_cell(bottleneck, bottleneck, 5, extra=extra, gpu=gpu) 
                    for _ in range(lstm_layers[1])])
        self.dense = nn.Linear(np.prod(bottleneck[1:]), 1)


    def forward(self, input, pred, hid_=None, extra=[]):
        '''
        TODO : make sure the right timestep is used when feeding extras. 
        check prednet to see how its done. 
        '''
        if hid_ is None : 
            hid_ = [self.clstms_input[i].init_hidden() for i in range(len(self.clstms_input))]

            # 1) iterate over timesteps of input
            for i in range(input.size(1)):
                x_t = input[:, i]
                extra_t = [ex[:, i, :] for ex in extra] if len(extra) > 0 else [] 
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
            extra_t = [ex[:, i+input.size(1), :] for ex in extra] if len(extra) > 0 else [] 
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



class CLSTM(nn.Module):
    
    def __init__(self, input_shape, output_shape, filter_size, num_layers):
	super(CLSTM, self).__init__()

	self.input_shape = input_shape
	self.output_shape = output_shape
	self.num_layers = num_layers
	cell_list = [CLSTM_cell(input_shape, output_shape, filter_size).cuda()]

	for _ in range(1, num_layers):
	    cell_list.append(CLSTM_cell(output_shape, output_shape, filter_size).cuda())

        self.cell_list = nn.ModuleList(cell_list)


    '''
    input        : tensor of shape (b_s, seq_len, C_inp, H, W)
    hidden_state : list of shape [(h_1, c_1), ..., (h_n, c_n)] for n layer CLSTM 

    returns
    next_hidden  : list of shape [(h_1, c_1), ..., (h_n, c_n)] for n layer CLSTM
    output       : tensor of shape (b_s, seq_len, C_hid, H, W) 
    '''
    def forward(self, input, hidden_state=None):
	if hidden_state is None : 
	     hidden_state = self.init_hidden()
	 
	input = input.transpose(0,1)
	current_input = input
	next_hidden = []
	seq_len = current_input.size(0)

	for l in range(self.num_layers):
	    h_l, c_l = hidden_state[l]   
	    layer_output = []

	    for t in range(seq_len):
		 h_l, c_l = self.cell_list[l](current_input[t,...], (h_l, c_l))
		 layer_output.append(h_l)

	    # save the last hidden state tuple (for possible hallucination)
	    next_hidden.append((h_l, c_l))            
	    # input of next layer is output of current layer
	    #current_input_old = torch.cat(layer_output, 0).view(current_input.size(0), *layer_output[0].size())
	    current_input = torch.stack(layer_output, 0)         

	return next_hidden, current_input.transpose(0,1)


    def init_hidden(self):
	init_states = []
	for i in range(self.num_layers):
	    init_states.append(self.cell_list[i].init_hidden())
	return init_states



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

