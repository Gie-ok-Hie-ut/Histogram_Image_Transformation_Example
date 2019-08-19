import torch
import torchvision
from torchvision import transforms
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
from PIL import Image
from skimage import color


img_torch = torch.rand(1,3,4,4)
print(img_torch)
arr = np.asarray(img_torch)
arr_new = [arr[0][1].flatten(), arr[0][2].flatten()]


print(arr[0][1].flatten())
print(arr[0][2].flatten())
print(ark)
print(arr[0][0][1])
#H,edges = np.histogramdd(arr_new, bins = [self.hist_l,224,224], range = ((-1,1),(-1,1),(-1,1)))
H,edges = np.histogramdd(arr_new, bins = [10,10], range = ((0,1),(0,1)))
H_torch = torch.from_numpy(H).float().cuda() #10/224/224
H_torch = H_torch.unsqueeze(0)
print(H_torch)
print(edges)
print(k)



def RGB2HSV_shift_LAB(I,shift): # shift value0 ~1
    # Get Original L in LAB, shift H in HSV

    # Get Original LAB
    lab_original = color.rgb2lab(I)
    l_original = (lab_original[:, :, 0] / 100.0)
    
    # Shift HSV
    hsv = color.rgb2hsv(I)
    h = ((hsv[:, :, 0] + shift))
    s = (hsv[:, :, 1])
    v = (hsv[:, :, 2])
    hsv2 = color.hsv2rgb(np.dstack([h, s, v]).astype(np.float64))

    # Merge (Original LAB, Shifted HSV)
    lab = color.rgb2lab(hsv2)
    l = l_original
    #l = (lab[:, :, 0] / 100.0)
    a = (lab[:, :, 1] + 86.1830297444) / (98.2330538631 + 86.1830297444) #* 255.0         # a component ranges from -127 to 127
    b = (lab[:, :, 2] + 107.857300207) / (94.4781222765 + 107.857300207) #* 255.0         # b component ranges from -127 to 127


    return np.dstack([l, a, b])

def LAB2RGB(I):
    # print(I)
    l = I[:, :, 0] / 255.0 * 100.0
    a = I[:, :, 1] / 255.0 * (98.2330538631 + 86.1830297444) - 86.1830297444
    b = I[:, :, 2] / 255.0 * (94.4781222765 + 107.857300207) - 107.857300207
    # print(np.dstack([l, a, b]))

    rgb = color.lab2rgb(np.dstack([l, a, b]).astype(np.float64))*255
    return rgb

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

####Image Read######
#shift = random.random() # 0~1
shift = 0
prep = transforms.Compose([transforms.Resize((256,256)),
                           transforms.Lambda(lambda img: RGB2HSV_shift_LAB(np.array(img),shift)),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5)),
                          ])

magnify = transforms.Compose([transforms.Resize((512,512)),
                             transforms.CenterCrop((256,256)),
    ])

str_img = 'lena1'
str_ext = '.jpg'

imgs = Image.open('./' + str_img + str_ext)
imgs_torch = prep(imgs)
imgs_torch = Variable(imgs_torch.unsqueeze(0).cuda())
imgs_torch1 = ( imgs_torch[:,0,:,:].unsqueeze(0).float().cpu() + 1 ) / 2 # 1/1/3/3
imgs_torch2 = ( imgs_torch[:,1,:,:].unsqueeze(0).float().cpu() + 1 ) / 2

# 1D Histogram
net = HistogramNet(64)
BIN = net.getBin()
net.getBiasedConv1().apply(net.init_biased_conv1)
#net.getBiasedConv1().apply(init_biased_conv1, num_bin=7)

t1 = net(imgs_torch1) #1/64/256/256
t2 = net(imgs_torch2)

# 2D Histogram
t11 = t1.repeat(1,BIN,1,1) #1/4096/256/256
t22 = t2.repeat(1,1,BIN,1).view(1,BIN*BIN,t2.size(2),t2.size(3)) #4/4096/256/256

pool = nn.AvgPool2d((t1.size(2),t2.size(3)))
hist2d = pool(t11*t22)
hist2d = hist2d.view(1,1,BIN,BIN)#*(3*3) # Size of image



#print(torch.sum(hist2d))

hist2d = hist2d * 20

#np.set_printoptions(threshold=np.nan)
#print(hist2d.detach().numpy())

hist2d_g = torch.zeros(1,2,BIN,BIN)
hist2d_all = torch.cat((hist2d,hist2d_g),1)
ups = nn.Upsample(scale_factor = 2 , mode = 'bilinear')
hist2d_all = ups(hist2d_all)

hist2d_all = hist2d_all[0].detach().cpu().float().numpy()
hist2d_np = hist2d[0].detach().cpu().float().numpy()
hist2d_np = hist2d_np[0]

#####calculate distance
#print(np.array(hist2d_np))
center = (BIN/2,BIN/2)
grid_x, grid_y = np.mgrid[0:BIN, 0:BIN]
grid_x = grid_x - center[0]
grid_y = grid_y - center[1]

dist = np.hypot(grid_x[hist2d_np > 0], grid_y[hist2d_np > 0]) # Get Distance above 0
weight = hist2d_np[hist2d_np>0] # Get Number of Bin above 0 (aligned with distances)
#weighted_dist = np.multiply(dist,weight) # Get Total distance of each bin
weighted_dist = weight
dist_mean = np.mean(weighted_dist)
print(dist_mean)

#image_numpy = hist2d_all * 255.0

#######Save Image
image_numpy = (np.transpose(hist2d_all,(1,2,0))) * 255.0

image_numpy = LAB2RGB(image_numpy)
image_numpy = image_numpy.astype(np.uint8)

image_pil = Image.fromarray(image_numpy)
image_pil.save('./' + str_img + '_hist' + str_ext)

