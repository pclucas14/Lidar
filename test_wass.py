from utils import * 

a = torch.cuda.FloatTensor(1, 100, 100).uniform_(-1,1)
b = torch.cuda.FloatTensor(1, 100, 100).uniform_(-1,1)
a, b = [Variable(x) for x in [a,b]]
ass = wasserstein_dist(a, b)
points = []
# get linear interpolations : 
for i in range(ass.size(0)):
    points.append(lin_inter(a[0][i], b[0][ass[i]]))


