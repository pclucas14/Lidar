import torch
import numpy as np 

x = torch.FloatTensor(32, 1000, 3)
index_first_dim = torch.LongTensor(np.arange(32))
index_second_dim = torch.LongTensor(np.arange(32))

print x[index_first_dim, index_second_dim].size()

x = torch.cuda.FloatTensor(32, 1000, 3)
index_first_dim = torch.cuda.LongTensor(np.arange(32))
index_second_dim = torch.cuda.LongTensor(np.arange(32))

print x[index_first_dim, index_second_dim].size()

