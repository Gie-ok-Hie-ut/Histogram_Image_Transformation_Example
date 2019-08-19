import torch
import torchvision
from torchvision import transforms
from torch.autograd import Variable
import numpy as np, numpy.random
import random
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

def RGB2LAB(I):
    # AB 98.2330538631 -86.1830297444 94.4781222765 -107.857300207
    lab = color.rgb2lab(I)
    l = (lab[:, :, 0] / 100.0)# * 255.0    # L component ranges from 0 to 100
    a = (lab[:, :, 1] + 86.1830297444) / (98.2330538631 + 86.1830297444) #* 255.0         # a component ranges from -127 to 127
    b = (lab[:, :, 2] + 107.857300207) / (94.4781222765 + 107.857300207) #* 255.0         # b component ranges from -127 to 127
    #l = (lab[:, :, 0] / 100.0) * 255.0    # L component ranges from 0 to 100
    #a = (lab[:, :, 1] + 86.1830297444) / (98.2330538631 + 86.1830297444) * 255.0         # a component ranges from -127 to 127
    #b = (lab[:, :, 2] + 107.857300207) / (94.4781222765 + 107.857300207) * 255.0         # b component ranges from -127 to 127
    return np.dstack([l, a, b])	

def RGB2HSV(I):
    hsv = color.rgb2hsv(I)
    #print(hsv[:, :, 0])
    h = ((hsv[:, :, 0] + 1.0 * 0.45) / 360.0 ) #* 255.0
    print(hsv[:, :, 0] + 0.5)
    s = (hsv[:, :, 1] / 100.0 ) #* 255.0
    v = (hsv[:, :, 2] / 100.0 ) #* 255.0
    return np.dstack([h, s, v])

def RGB2HSV_shift(I,shift): # shift value0 ~1
    hsv = color.rgb2hsv(I)
    #print(hsv[:, :, 0])
    h = ((hsv[:, :, 0] + shift) / 360.0 ) #* 255.0
    s = (hsv[:, :, 1] / 100.0 ) #* 255.0
    v = (hsv[:, :, 2] / 100.0 ) #* 255.0
    return np.dstack([h, s, v])

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

def HSV2RGB(I):

    h = I[:, :, 0] / 255.0 * 360.0
    s = I[:, :, 1] / 255.0 * 100.0
    v = I[:, :, 2] / 255.0 * 100.0
    # print(np.dstack([l, a, b]))

    hsv = color.hsv2rgb(np.dstack([h, s, v]).astype(np.float64))*255
    return hsv

def LAB2RGB(I):
    # print(I)
    l = I[:, :, 0] / 255.0 * 100.0
    a = I[:, :, 1] / 255.0 * (98.2330538631 + 86.1830297444) - 86.1830297444
    b = I[:, :, 2] / 255.0 * (94.4781222765 + 107.857300207) - 107.857300207
    # print(np.dstack([l, a, b]))

    rgb = color.lab2rgb(np.dstack([l, a, b]).astype(np.float64))*255
    return rgb



def tensor2im(image_tensor,toRGB = False,imtype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
        
    # Additional
    if toRGB == True:
        image_numpy = LAB2RGB(image_numpy) # for Lab to RGB image
        #image_numpy = HSV2RGB(image_numpy) # LAB to RGB
    #image_numpy = np.clip(image_numpy,0,255)
    return image_numpy.astype(imtype)

shift = random.random() # 0~1
print(shift)
prep = transforms.Compose([
						   transforms.Lambda(lambda img: RGB2HSV_shift_LAB(np.array(img),shift)),
                           transforms.ToTensor(),
						   transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5)),

                           #transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to BGR
                           #transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961], #subtract imagenet mean
                           #                     std=[1,1,1]),
                           #transforms.Lambda(lambda x: x.mul_(255)),
                          ])

imgs = Image.open('./home1.jpg')

#### RGB to LAB
imgs_torch = prep(imgs)
#print(imgs_torch)
imgs_torch = Variable(imgs_torch.unsqueeze(0).cuda())

#print(imgs_torch)
#getHistogram(imgs_torch)
#### Torch to Numpy
image_numpy = imgs_torch[0].cpu().float().numpy()

#image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0 # Normalized
image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0 # Normalized
#image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) # Not Normalized
#print(image_numpy)
#### LAB to RGB
#image_numpy = LAB2RGB(image_numpy)
#image_numpy = image_numpy.astype(np.uint8)

#####Save Image
#image_pil = Image.fromarray(image_numpy)
#image_pil.save('./home12_rand2.jpg')




############################################Palette Generator
# Setting
k = random.randrange(1,10)
width = 256

# Length preprocess
divided_length = (np.random.dirichlet(np.ones(k),size = 1) * width ).astype(int).tolist()[0]

if sum(divided_length) < width:
    gap = width - sum(divided_length)
    divided_length[k-1] = divided_length[k-1] + gap

# Color preprocess
color = torch.rand(k,3)

# Stack Color
img = torch.zeros(1,3,width,width)
last = 0
for i in range(0,k):
    size_temp = torch.ones(1,3,width,divided_length[i])
    if divided_length[i] != 0:
        color_temp = color[i].unsqueeze(0).unsqueeze(0).unsqueeze(0).permute(0,3,1,2)
        img_temp = size_temp * color_temp

        img[:,:,:,last:(last+divided_length[i])] = img_temp
        last = last + divided_length[i] 







imgs_torch = img
image_numpy = imgs_torch[0].cpu().float().numpy()

#image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0 # Normalized
image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0 # Normalized
#image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) # Not Normalized
#print(image_numpy)
#### LAB to RGB
#image_numpy = LAB2RGB(image_numpy)
image_numpy = image_numpy.astype(np.uint8)

#####Save Image
image_pil = Image.fromarray(image_numpy)
image_pil.save('./timeout.jpg')

















