import os
import glob
import torch
from torch import nn
from torchvision.models import vgg19_bn
from torchvision import transforms
from PIL import Image
import tifffile as tiff
import numpy as np

# Check if a GPU is available and if not, use a CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Define the VGG19 model
class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        # Define each block separately
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        # Pass the input through each block and store the feature maps
        feature_map1 = self.block1(x)
        feature_map2 = self.block2(feature_map1)
        feature_map3 = self.block3(feature_map2)
        feature_map4 = self.block4(feature_map3)
        feature_map5 = self.block5(feature_map4)
        # Return all the feature maps
        return feature_map1, feature_map2, feature_map3, feature_map4, feature_map5

# Define the upsampling part of the network
class Upsample(nn.Module):
    def __init__(self):
        super(Upsample, self).__init__()
        # Define upsampling blocks for each feature map
        self.upsample1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.upsample2 = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=4)
        self.upsample3 = nn.ConvTranspose2d(256, 256, kernel_size=8, stride=8)
        self.upsample4 = nn.ConvTranspose2d(512, 512, kernel_size=16, stride=16)
        self.upsample5 = nn.ConvTranspose2d(512, 512, kernel_size=32, stride=32)
        # Define the final layer that combines the upsampled feature maps
        self.final = nn.Sequential(
            nn.Conv2d(64 + 128 + 256 + 512 + 512, 1, kernel_size=1),  # Combines the upsampled feature maps
            nn.Sigmoid()  # Produces the final saliency map
        )

    def forward(self, feature_map1, feature_map2, feature_map3, feature_map4, feature_map5):
        # Upsample each feature map
        upsampled1 = self.upsample1(feature_map1)
        upsampled2 = self.upsample2(feature_map2)
        upsampled3 = self.upsample3(feature_map3)
        upsampled4 = self.upsample4(feature_map4)
        upsampled5 = self.upsample5(feature_map5)
        # Concatenate the upsampled feature maps
        concatenated = torch.cat((upsampled1, upsampled2, upsampled3, upsampled4, upsampled5), dim=1)
        # Pass the concatenated feature maps through the final layer to get the saliency map
        saliency_map = self.final(concatenated)
        return saliency_map


# Load pre-trained VGG19 model
model = VGG()
model=model.to(device)

# Load weights from PytorchHub of model vgg19_bn
weights = torch.load('vgg19_bn-c79401a0.pth')

model.eval()  # Set the model to evaluation mode

# Remove all layers after the convolutional layers to get the feature map
model.classifier = nn.Identity()
model.avgpool = nn.Identity()

# Initialize the upsampling network
upsample = Upsample().to(device)

# Specify the folder containing the 3D tiff stacks
folder = 'Patches_Tumeur_ech11'

# Specify the path to the Desktop
desktop_path = os.path.join(os.path.expanduser("~"), 'Desktop')

# Create the "Saliency" folder on the Desktop if it doesn't exist
saliency_folder ='Saliency'
if not os.path.exists(saliency_folder):
    os.makedirs(saliency_folder)

# Process each 3D tiff stack in the folder
for filepath in glob.glob(os.path.join(folder, '*.tif')):
    # Load the 3D tiff stack
    stack = tiff.imread(filepath)

    # Initialize an empty list to store the saliency maps
    saliency_maps = []

    # Process each slice of the stack
    for i in range(stack.shape[0]):
        slice = stack[i, :, :]

        # Convert the slice to a PIL image, convert it to RGB, resize it to 224x224 (required by VGG19), and normalize it
        slice = Image.fromarray(slice).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        slice = transform(slice).unsqueeze(0)  # Add an extra dimension for the batch size

        # Pass the slice through the VGG19 model to get the feature maps
        slice = slice.to(device)
        feature_maps = model(slice)

        # Pass the feature maps through the upsampling network to get the saliency map
        saliency_map = upsample(*feature_maps)

        # Add the saliency map to the list
        saliency_maps.append(saliency_map.detach().cpu().numpy())

    # Stack the saliency maps into a 3D tiff stack
    saliency_stack = np.stack(saliency_maps)

    # Save the 3D tiff stack in the "Saliency" folder on the Desktop
    # Use the original file name with a "_saliency" suffix
    base_name = os.path.basename(filepath)
    base_name = os.path.splitext(base_name)[0] + '_saliency.tif'
    tiff.imwrite(os.path.join(saliency_folder, base_name), saliency_stack)

import numpy as np
import re

# Load the existing 3D tiff stacks from the Saliency folder
saliency_folder = 'Saliency'
stacks = []

# Load and append each 3D tiff stack in the saliency folder
for filepath in sorted(glob.glob(os.path.join(saliency_folder, '*.tif')), key=lambda x: int(re.findall(r'\d+', x)[0])):
    stack = tiff.imread(filepath)
    stacks.append(stack)

# Concatenate the 3D tiff stacks along the z-axis
big_stack = np.concatenate(stacks, axis=0)

# Save the big 3D tiff stack in the Saliency folder
tiff.imwrite(os.path.join(saliency_folder, 'big_stack.tif'), big_stack)



