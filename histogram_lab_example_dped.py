import torch
import torchvision
from torchvision import transforms
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
from PIL import Image
from skimage import color
import os

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

def getDistance(str_input, dist_list):
    imgs = Image.open(str_input)
    # Preprocessing
    imgs_torch = prep(imgs)
    imgs_torch = Variable(imgs_torch.unsqueeze(0).cuda())
    imgs_torch1 = ( imgs_torch[:,0,:,:].unsqueeze(0).float().cpu() + 1 ) / 2 # 1/1/3/3
    imgs_torch2 = ( imgs_torch[:,1,:,:].unsqueeze(0).float().cpu() + 1 ) / 2


    # 1D Histogram
    net = HistogramNet(64)
    BIN = net.getBin()
    net.getBiasedConv1().apply(net.init_biased_conv1)
    
    t1 = net(imgs_torch1) #1/64/256/256
    t2 = net(imgs_torch2)

    pool0 = nn.AvgPool2d(t1.size(2),t1.size(3)) # 1D histogram
    t1d = pool0(t1).detach().numpy() #
    t2d = pool0(t2).detach().numpy()

    # Calc center of 1D Histogram
    grid1d = np.mgrid[1:BIN+1]
    cx = np.sum(t1d.flatten()*grid1d)
    cy = np.sum(t2d.flatten()*grid1d)
    
    # 2D Histogram
    t11 = t1.repeat(1,BIN,1,1) #1/4096/256/256
    t22 = t2.repeat(1,1,BIN,1).view(1,BIN*BIN,t2.size(2),t2.size(3)) #4/4096/256/256
    
    pool = nn.AvgPool2d((t1.size(2),t2.size(3)))
    hist2d = pool(t11*t22)
    hist2d = hist2d.view(1,1,BIN,BIN)
    
    hist2d = hist2d * 20
    
    # Post-processsing
    hist2d_g = torch.zeros(1,2,BIN,BIN)
    hist2d_all = torch.cat((hist2d,hist2d_g),1) # make 3 channel
    ups = nn.Upsample(scale_factor = 2 , mode = 'bilinear')
    hist2d_all = ups(hist2d_all) # make it bigger
    
    hist2d_all = hist2d_all[0].detach().cpu().float().numpy()
    hist2d_np = hist2d[0].detach().cpu().float().numpy()
    hist2d_np = hist2d_np[0]
    
    # Calculate distance
    #cx = BIN/2
    #cy = BIN/2 # gray center
    grid_y, grid_x = np.mgrid[0:BIN, 0:BIN]
    grid_x = grid_x - (cx-1)
    grid_y = grid_y - (cy-1)

    dist = np.hypot(grid_x[hist2d_np > 0], grid_y[hist2d_np > 0]) # Get Distance above 0
    weight = hist2d_np[hist2d_np>0] # Get Number of Bin above 0 (aligned with distances)
    
    #weighted_dist = np.multiply(dist,weight) # Get Total distance of each bin (may not be aligned)
    weighted_dist = dist
    dist_mean = np.mean(weighted_dist)

    #print("MEAN")
    print(dist_mean)

    dist_list.append(dist_mean)
    #print(dist_list)

    
    ######Save Image
    #image_numpy = (np.transpose(hist2d_all,(1,2,0))) * 255.0
    #image_numpy = LAB2RGB(image_numpy)
    #image_numpy = image_numpy.astype(np.uint8)
    #image_pil = Image.fromarray(image_numpy)
    #if str_input.find('canon') > 0 :
    #    image_pil.save('./sony/' + str(i) + '_hist_cn' + str_ext)
    #else:
    #    image_pil.save('./sony/' + str(i) + '_hist' + str_ext)
        

    return dist_mean
####Image Read######
#shift = random.random() # 0~1

shift = 0
prep = transforms.Compose([transforms.Resize((256,256)),
                           transforms.Lambda(lambda img: RGB2HSV_shift_LAB(np.array(img),shift)),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5)),
                          ])


str_bad = '/root/Mango/Common/Dataset/dped/sony/training_data/sony/'
str_good = '/root/Mango/Common/Dataset/dped/sony/training_data/canon/'
str_ext = '.jpg'
dist_list_bad = []
dist_list_good = []
good_is_wider=0
good_is_narrower=0

for i in range(1,100 + 1): 
    print(' ')

    distance1 = getDistance(str_bad + str(i) + str_ext, dist_list_bad)
    distance2 = getDistance(str_good + str(i) + str_ext, dist_list_good)

    if (distance1 < distance2): #Good
        good_is_wider= good_is_wider+1
    else:
        good_is_narrower= good_is_narrower+1

    print('Processing '+str(i))
    print(good_is_wider)
    print(good_is_narrower)
    #print(end)


print("[dist_list_bad]")
print(dist_list_bad)
print(np.mean(dist_list_bad))
print(np.var(dist_list_bad))

print("[dist_list_good]")
print(dist_list_good)
print(np.mean(dist_list_good))
print(np.var(dist_list_good))
