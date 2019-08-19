import torch
import torchvision
from torchvision import transforms
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
from PIL import Image
from skimage import color


#from torch.utils.serialization import load_lua
import torchfile






#def loadSegmentation(self,address):
#    x = load_lua(address[0],unknown_classes=True) #1/8/392/496
#    return x.cuda()
#segment = Variable(loadSegmentation('./a0001-jmac_DSC1459_bic.t7'))
temp_img = torch.rand(1,3,256,256).cuda()

# Load t7 file
segment = torchfile.load('./a0001-jmac_DSC1459_bic.t7')
segment_torch = torch.from_numpy(segment).cuda()

# Get Indicies
values, indicies = torch.max(segment_torch, 1) # Max Label
print(indicies==0)


# Expand Indicies
indicies_expand = torch.zeros(8,256,256)

indicies_expand[0:1,:,:] = (indicies == 0).float()
indicies_expand[1:2,:,:] = (indicies == 1).float()

print(indicies_expand[0:1,:,:])
print(Kr)

indicies0 = ((indicies - 0 ) == 0).float()
indicies1 = ((indicies - 1 ) == 0).float()
print(indicies0)
print(indicies_expand)
print(more)

# Count PixelNum
num0 = torch.sum(indicies0)
num1 = torch.sum(indicies1)

# Get Image Segmentwise
temp_img0 = torch.mul(temp_img,indicies0) 
print(temp_img0)

# Calculate Histogram Each Segment
# getHistogram2d_np
#
#
#
#

# Calculate Enc for each segment
#
#
#
#
#
enc_ex = torch.rand(8,64,1,1) # Strange Form


########### Shuffle Segment
segment2 = torchfile.load('./a0001-jmac_DSC1459_bic.t7')
segment_torch2 = torch.from_numpy(segment2).cuda()

values2, indicies2 = torch.max(segment_torch2, 1) # Max Label
#print(enc_ex)
#print(enc_ex[0:1,:,:,:])
#print(indicies2)


final_result = torch.zeros(1,64,256,256).cuda() # Basically Should be global feature for fear of absent object


# Ingredient
final_result
indicies2
enc_ex

final_result[:,:,indicies2.squeeze(0)==0] = enc_ex[0:1,:,:,:].repeat(1,1,final_result[:,:,indicies2.squeeze(0)==0].size(2),1).squeeze(0).permute(2,0,1).cuda()
final_result[:,:,indicies2.squeeze(0)==1] = enc_ex[1:2,:,:,:].repeat(1,1,final_result[:,:,indicies2.squeeze(0)==1].size(2),1).squeeze(0).permute(2,0,1).cuda()

print(indicies2.squeeze(0)==0)
print(final_result[:,:,indicies2.squeeze(0)==0])
print(final_result[:,:,indicies2.squeeze(0)==0].size(2))
print(enc_ex[0:1,:,:,:].repeat(1,1,final_result[:,:,indicies2.squeeze(0)==0].size(2),1).squeeze(0).permute(2,0,1).shape)
print(final_result)
