#https://github.com/leongatys/PytorchNeuralStyleTransfer/blob/master/NeuralStyleTransfer.ipynb

import time
import os 
image_dir = os.getcwd() + '/Images/'
model_dir = os.getcwd() + '/Models/'

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np

import torchvision
from torchvision import transforms

from PIL import Image
from collections import OrderedDict

import totalvariation


class VGG(nn.Module):
    def __init__(self, pool='max'):
        super(VGG, self).__init__()
        #vgg modules
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        if pool == 'max':
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        elif pool == 'avg':
            self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)
            
    def forward(self, x, out_keys):

        out = {}
        out['r11'] = F.relu(self.conv1_1(x))
        out['r12'] = F.relu(self.conv1_2(out['r11']))
        out['p1'] = self.pool1(out['r12'])
        out['r21'] = F.relu(self.conv2_1(out['p1']))
        out['r22'] = F.relu(self.conv2_2(out['r21']))
        out['p2'] = self.pool2(out['r22'])
        out['r31'] = F.relu(self.conv3_1(out['p2']))
        out['r32'] = F.relu(self.conv3_2(out['r31']))
        out['r33'] = F.relu(self.conv3_3(out['r32']))
        out['r34'] = F.relu(self.conv3_4(out['r33']))
        out['p3'] = self.pool3(out['r34'])
        out['r41'] = F.relu(self.conv4_1(out['p3']))
        out['r42'] = F.relu(self.conv4_2(out['r41']))
        out['r43'] = F.relu(self.conv4_3(out['r42']))
        out['r44'] = F.relu(self.conv4_4(out['r43']))
        out['p4'] = self.pool4(out['r44'])
        out['r51'] = F.relu(self.conv5_1(out['p4']))
        out['r52'] = F.relu(self.conv5_2(out['r51']))
        out['r53'] = F.relu(self.conv5_3(out['r52']))
        out['r54'] = F.relu(self.conv5_4(out['r53']))
        out['p5'] = self.pool5(out['r54'])
        return [out[key] for key in out_keys]

# gram matrix and loss
class GramMatrix(nn.Module):
    def forward(self, input):
        b,c,h,w = input.size()
        F = input.view(b, c, h*w)
        G = torch.bmm(F, F.transpose(1,2)) 
        G.div_(h*w)
        return G

class GramMSELoss(nn.Module):
    def forward(self, input, target):
        out = nn.MSELoss()(GramMatrix()(input), target)
        return(out)


# pre and post processing for images
img_size = 224
prep = transforms.Compose([transforms.Resize(img_size),
                           transforms.ToTensor(),
                           transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to BGR
                           transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961], #subtract imagenet mean
                                                std=[1,1,1]),
                           transforms.Lambda(lambda x: x.mul_(255)),
                          ])
postpa = transforms.Compose([transforms.Lambda(lambda x: x.mul_(1./255)),
                           transforms.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961], #add imagenet mean
                                                std=[1,1,1]),
                           transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to RGB
                           ])
postpb = transforms.Compose([transforms.ToPILImage()])
def postp(tensor): # to clip results in the range [0,1]
    t = postpa(tensor)
    t[t>1] = 1    
    t[t<0] = 0
    img = postpb(t)
    return img


string_vgg_color = '../checkpoints/colorclassify_pretrained_norm_vgg19_7syn/10_net_G_A.pth'
string_vgg_original = '../vgg_original.pth'
string_centroid_r21 = '../centroid/Syn7_ep10/centroid_Syn_night_r21.pt'
string_centroid_r31 = '../centroid/Syn7_ep10/centroid_Syn_night_r31.pt'
string_centroid_r41 = '../centroid/Syn7_ep10/centroid_Syn_night_r41.pt'
string_centroid_r51 = '../centroid/Syn7_ep10/centroid_Syn_night_r51.pt'
string_img_name = './Syn7_ep10_percep_r41_color_r4151/result_night_200_tv3.jpg'



#get network
vgg_color = VGG()
#vgg_color.load_state_dict(torch.load('../135_net_G_A.pth')) # For the BCE
vgg_color.load_state_dict(torch.load(string_vgg_color)) # For the BCEN
#vgg_color.load_state_dict(torch.load('../vgg_original.pth'))
vgg_color.cuda()

vgg_percep = VGG()
vgg_percep.load_state_dict(torch.load(string_vgg_original))
vgg_percep.cuda()

for param in vgg_color.parameters():
    param.requires_grad = False
for param in vgg_percep.parameters():
    param.requires_grad = False


# Get Feature
#centroid_r21=torch.load('../centroid/BCEN_10/centroid_BCEN_N_r21.pt')
#centroid_r31=torch.load('../centroid/BCEN_10/centroid_BCEN_N_r31.pt')
#centroid_r41=torch.load('../centroid/BCEN_10/centroid_BCEN_N_r41.pt')
#centroid_r51=torch.load('../centroid/BCEN_10/centroid_BCEN_N_r51.pt') 
#centroid_r21=torch.load('../centroid/VGG/centroid_vgg_N_r21.pt')
#centroid_r31=torch.load('../centroid/VGG/centroid_vgg_N_r31.pt')
#centroid_r41=torch.load('../centroid/VGG/centroid_vgg_N_r41.pt')
#centroid_r51=torch.load('../centroid/VGG/centroid_vgg_N_r51.pt') 
centroid_r21=torch.load(string_centroid_r21)
centroid_r31=torch.load(string_centroid_r31)
centroid_r41=torch.load(string_centroid_r41)
centroid_r51=torch.load(string_centroid_r51)


#load images, ordered as [style_image, content_image]
# We will change 'content'
img_names = ['./lena1.jpg', './home1.jpg']
imgs = [Image.open(name) for i,name in enumerate(img_names)]
imgs_torch = [prep(img) for img in imgs]
imgs_torch = [Variable(img.unsqueeze(0).cuda()) for img in imgs_torch]

style_image, content_image = imgs_torch

# opt_img = Variable(torch.randn(content_image.size()).type_as(content_image.data), requires_grad=True) #random init
opt_img = Variable(content_image.data.clone(), requires_grad=True)


#define layers, loss functions, weights and compute optimization targets
content_layers = ['r41'] 
color_layers = ['r41','r51']


loss_fns_content = [nn.MSELoss()] * len(content_layers)
loss_fns_color = [nn.MSELoss()] * len(color_layers)

loss_fns_content = [loss_fn.cuda() for loss_fn in loss_fns_content]
loss_fns_color = [loss_fn.cuda() for loss_fn in loss_fns_color]


#these are good weights settings:
content_weights = [1e3/n**2 for n in [64,128,256,512,512]]
color_weights = [1e3/n**2 for n in [64,128,256,512,512]]



#compute optimization targets
content_targets = [A.detach() for A in vgg_percep(content_image, content_layers)]
color_targets = [centroid_r41,centroid_r51]

#run style transfer
max_iter = 1000
show_iter = 50
optimizer = optim.LBFGS([opt_img]);
n_iter=[0]


criterionMSE = torch.nn.MSELoss()
tv = totalvariation.TVLoss()
while n_iter[0] <= max_iter:

    def closure():
        optimizer.zero_grad()

        # inference and get specific features
        out_content = vgg_percep(opt_img,content_layers)
        out_color = vgg_color(opt_img, color_layers) 
        
        
        # weights  : [0.244140625, 0.06103515625, 0.0152587890625, 0.003814697265625, 0.003814697265625, 1.0]
        # loss_fns1 : [GramMSELoss(), GramMSELoss(), GramMSELoss(), GramMSELoss(), GramMSELoss(), MSELoss()]
        # targets  : 6 features

        content_losses = [ content_weights[b] * loss_fns_content[b](B, content_targets[b]) for b,B in enumerate(out_content)]
        color_losses = [ color_weights[a] * loss_fns_color[a](A, color_targets[a]) for a,A in enumerate(out_color)]   



        #tv_loss, _ = totalvariation.tv_norm(opt_img.data.cpu().numpy())

        tv_loss = tv(opt_img)



        loss = sum(content_losses) + sum(color_losses) * 20 + tv_loss#torch.from_numpy(np.array(tv_loss)) * 0.05
        loss.backward()


        n_iter[0]+=1

        #print loss
        if n_iter[0]%show_iter == (show_iter-1):
            print('Iteration: %d, loss: %f'%(n_iter[0]+1, loss.data[0]))
            #print([loss_layers1[li] + ': ' +  str(l.data[0]) for li,l in enumerate(layer_losses)]) #loss of each layer
        return loss
    
    optimizer.step(closure)
    
#display result
out_img = postp(opt_img.data[0].cpu().squeeze())
out_img.save(string_img_name)