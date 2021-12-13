import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.models as models
import random
import time

from transfer_network import StyleTransfer
from dataset import get_content_dataset, get_painting_dataset, get_avg_dataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# FOR THE PICASSO ONES, COMMENT OUT THE NEW LAYERS IN THE ARCHITECTURE
STYLE_METHOD = 'random'
ARTIST = 'Leonardo_da_Vinci'

model_dir = 'models/' + ARTIST + '/' + STYLE_METHOD + '/'

net = StyleTransfer().double()
net.load_state_dict(torch.load(model_dir + 'transfer_17-40_32.pth'), strict=False)  # 'Pablo_Picasso_transfer2_280.pth'
net = net.to(device)
net.eval()

input_img = cv2.imread('dancing.jpg').astype(float).transpose(2, 0, 1)
input_tensor = torch.from_numpy(input_img).to(device).unsqueeze(0)

style_img = cv2.imread(model_dir + 'style.jpg').astype(float).transpose(2, 0, 1)
style_img = style_img[[2, 1, 0]].transpose(1, 2, 0).clip(0, 255).astype('uint8')
content_img = input_img[[2, 1, 0]].transpose(1, 2, 0).clip(0, 255).astype('uint8')

with torch.no_grad():
    out_tensor = net(input_tensor)
out_img = out_tensor.detach().cpu().squeeze().numpy()[[2, 1, 0]].transpose(1, 2, 0).clip(0, 255).astype('uint8')

plt.close('all')
fig = plt.figure(figsize=(18, 5))
fig.add_subplot(1, 3, 1)
plt.imshow(content_img, interpolation='nearest', aspect='auto')
plt.title('Content')
plt.pause(0.001)

fig.add_subplot(1, 3, 2)
plt.imshow(style_img, interpolation='nearest', aspect='auto')
plt.title('Style')
plt.pause(0.001)

fig.add_subplot(1, 3, 3)
plt.imshow(out_img, interpolation='nearest', aspect='auto')
plt.title('Transformed')
plt.pause(0.001)
plt.show()
