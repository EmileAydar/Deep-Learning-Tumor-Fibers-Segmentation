# Emile Aydar
# Fiber Detection in Cold-Plasma treated lung tumors
# LPP/ISIR || Ecole Polytechnique/Sorbonne Université, 2023

# VGG-19bn-backboned neural network for image saliency detection 

import os
import glob
import torch
from torch import nn
from torchvision.models import vgg19_bn
import torchvision.transforms as transforms
from PIL import Image
import tifffile as tiff
import numpy as np

##########################################################GPU SETTER##########################################################
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#################################SALIENCY DETECTION WITH VGG-19BN BACKBONED NEURAL NETWORK####################################
class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.block5[-3].register_forward_hook(self.forward_hook)
        self.block5[-3].register_full_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        self.feature_maps = output.detach()

    def backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].clone()

    def forward(self, x):
        feature_map1 = self.block1(x)
        feature_map2 = self.block2(feature_map1)
        feature_map3 = self.block3(feature_map2)
        feature_map4 = self.block4(feature_map3)
        self.output = self.block5(feature_map4)  # save the output here

        return feature_map1, feature_map2, feature_map3, feature_map4, self.output


class Upsample(nn.Module):
    def __init__(self):
        super(Upsample, self).__init__()

        self.upsample1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 64, kernel_size=3, padding=1)
        )
        self.upsample2 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 128, kernel_size=3, padding=1)
        )
        self.upsample3 = nn.Sequential(
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 256, kernel_size=3, padding=1)
        )
        self.upsample4 = nn.Sequential(
            nn.Upsample(scale_factor=16, mode='bilinear', align_corners=False),
            nn.Conv2d(512, 512, kernel_size=3, padding=1)
        )
        self.upsample5 = nn.Sequential(
            nn.Upsample(scale_factor=32, mode='bilinear', align_corners=False),
            nn.Conv2d(512, 512, kernel_size=3, padding=1)
        )

        self.final = nn.Sequential(
            nn.Conv2d(64 + 128 + 256 + 512 + 512, 1, kernel_size=1),  # combines the upsampled feature maps
            nn.Sigmoid() 
        )

        # optional gaussian blur (has no use tbh but why not :)
        #self.gaussian_blur = transforms.GaussianBlur(5, sigma=(0.1, 2.0))

    def forward(self, feature_map1, feature_map2, feature_map3, feature_map4, feature_map5):
        upsampled1 = self.upsample1(feature_map1)
        upsampled2 = self.upsample2(feature_map2)
        upsampled3 = self.upsample3(feature_map3)
        upsampled4 = self.upsample4(feature_map4)
        upsampled5 = self.upsample5(feature_map5)

        concatenated = torch.cat((upsampled1, upsampled2, upsampled3, upsampled4, upsampled5), dim=1)

        saliency_map = self.final(concatenated)

        #saliency_map = self.gaussian_blur(saliency_map)
        return saliency_map

#########################################SALIENCY DETECTION###################################################################
model = VGG()
model = model.to(device)

weights = torch.load('vgg19_bn-c79401a0.pth')

model.eval() 

model.classifier = nn.Identity()
model.avgpool = nn.Identity()

upsample = Upsample().to(device)

folder = 'bigstack'

desktop_path = os.path.join(os.path.expanduser("~"), 'Desktop')

saliency_folder = 'newSaliency'
if not os.path.exists(saliency_folder):
    os.makedirs(saliency_folder)

for filepath in glob.glob(os.path.join(folder, '*.tif')):
    stack = tiff.imread(filepath)

    saliency_maps = []

    for i in range(stack.shape[0]):
        slice = stack[i, :, :]

        slice = Image.fromarray(slice).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        slice = transform(slice).unsqueeze(0)  # add an extra dimension for the batch size

        slice = slice.to(device)
        feature_map1, feature_map2, feature_map3, feature_map4, output = model(slice)

        torch.autograd.set_grad_enabled(True)

        model.zero_grad()
        output.backward(torch.ones_like(output)) 

        feature_maps = model.feature_maps
        gradients = model.gradients

        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

        for i in range(512):
            feature_maps[:, i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(feature_maps, dim=1).squeeze()
        heatmap = torch.relu(heatmap)

        heatmap /= torch.max(heatmap)

        saliency_map = upsample(feature_map1, feature_map2, feature_map3, feature_map4, output)

        saliency_maps.append(saliency_map.detach().cpu().numpy())

    saliency_stack = np.stack(saliency_maps)


    base_name = os.path.basename(filepath)
    base_name = os.path.splitext(base_name)[0] + '_saliency.tif'
    tiff.imwrite(os.path.join(saliency_folder, base_name), saliency_stack)
