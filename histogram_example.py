import torch
import torchvision
from torchvision import transforms
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
from PIL import Image
from skimage import color

def getHistogram(imgs_torch):
	#arr = np.asarray(imgs_torch)#.convert("RGB"))
	#print(arr.size)
	#arr = np.array(imgs_torch.getdata()).reshape(imgs_torch.size[0], imgs_torch.size[1], 3)
	arr = np.asarray(imgs_torch)
	#print(arr)
	#print(arr[0][0][0])
	#arr_new = [arr[:,:,0][0], arr[:,:,1][0], arr[:,:,2][0]]
	arr_new = [arr[0][0][0], arr[0][0][1], arr[0][0][2]]
	H,edges = np.histogramdd(arr_new, bins = [10,200,200], range = ((0,255),(0,255),(0,255)))
	H_torch = torch.from_numpy(H).float()
	print(H_torch)
	print(edges) # edges include bin position


class HistogramNet(nn.Module):
    def __init__(self,bin_num):
        super(HistogramNet,self).__init__()
        self.bin_num = bin_num
        self.LHConv_1 = BiasedConv1(1,bin_num)
        self.relu = nn.ReLU(True)

    def forward(self,input):
        a1 = self.LHConv_1(input)
        a2 = torch.abs(a1)
        a3 = 1- a2*(self.bin_num-1)
        a4 = self.relu(a3)
        return a4

    def getBiasedConv1(self):
        return self.LHConv_1

    def getBin(self):
        return self.bin_num

    def init_biased_conv1(self,m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            m.bias.data = -torch.arange(0,1,1/(self.bin_num-1))
            m.weight.data = torch.ones(self.bin_num,1,1,1)

class BiasedConv1(nn.Module):
    def __init__(self,dim_in,dim_out):
        super(BiasedConv1, self).__init__()
        model = []
        model += [nn.Conv2d(dim_in,dim_out,kernel_size=1,padding=0,stride=1,bias=True),]
        self.model = nn.Sequential(*model)

    def forward(self,input):
        a = self.model(input)
        return a

        
# Tensor Preparation
#imgs_torch1 = torch.rand(1,1,2,2)
#imgs_torch2 = torch.rand(1,1,2,2)
imgs_torch1 = torch.tensor([[1.0,0.4, 0.2],[0.2,0.8, 0.2], [0.2,0.8, 0.2]]).unsqueeze(0).unsqueeze(0)
imgs_torch2 = torch.tensor([[0.6,0.0, 0.2],[0.2,0.6, 0.4], [0.2,0.8, 0.2]]).unsqueeze(0).unsqueeze(0)
print(imgs_torch1)
print(imgs_torch2)

# 1D Histogram
net = HistogramNet(6)
BIN = net.getBin()
net.getBiasedConv1().apply(net.init_biased_conv1)
#net.getBiasedConv1().apply(init_biased_conv1, num_bin=7)

t1 = net(imgs_torch1) #1/6/3/3
t2 = net(imgs_torch2)
#print(t1)

# 2D Histogram
t11 = t1.repeat(1,BIN,1,1) #1/36/3/3
t22 = t2.repeat(1,1,BIN,1).view(1,BIN*BIN,3,3) #1/36/3/3

pool = nn.AvgPool2d(3)
hist2d = pool(t11*t22)
hist2d = hist2d.view(1,1,BIN,BIN)#*(3*3) # Size of image


# Final
#print(hist2d)
# X-Y Graph
hist1 = pool(t1)
hist2 = hist1.view(1,1,1,BIN).repeat(1,1,BIN,1)
print(hist1.view(1,1,1,BIN))
hist3 = net(hist2)
pool2 = nn.AvgPool2d((6,1))
hist4 = pool2(hist3)
hist4 = hist4.view(1,1,6,6)
print(hist4)


