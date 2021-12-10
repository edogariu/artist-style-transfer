import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.models as models
import torchvision.transforms as transforms
from dataset import get_dataset
from PIL import Image
import random

# Acknowledgements:
# Much of this code (dealing with the pre-trained VGG and crafting the content and first style loss functions via the
# VGG is adapted from this tutorial: https://pytorch.org/tutorials/advanced/neural_style_tutorial.html and
# this pre-written code: https://github.com/leongatys/PytorchNeuralStyleTransfer/blob/master/NeuralStyleTransfer.ipynb

# ------------------------------------------------------------------------------------------------------------------
# HYPERPARAMETERS
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}

# imsize = 512 if torch.cuda.is_available() else 128  # use small size if no gpu

NUM_EPOCHS = 100
CONTENT_IMAGE_PATH = "dancing.jpg"
STYLE_IMAGE_PATH = "picasso.jpg"
BATCH_SIZE = 1
CONTENT_WEIGHT = 17  # 17
STYLE_WEIGHT = 50  # 25
ADAM_LR = 0.002
SEED = 0
MODEL_SAVE = 100  # How often to save model during training

# desired depth layers to compute content/style losses :
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']


# -------------------------------------------------------------------------------------------------------------------
# TRANSFORMER
class StyleTransfer(nn.Module):
    def __init__(self):
        super(StyleTransfer, self).__init__()
        self.ConvBlock = nn.Sequential(
            ConvLayer(3, 32, 9, 1),
            nn.ReLU(),
            ConvLayer(32, 64, 3, 2),
            nn.ReLU(),
            ConvLayer(64, 128, 3, 2),
            nn.ReLU()
        )
        self.ResidualBlock = nn.Sequential(
            ResidualLayer(128, 3),
            ResidualLayer(128, 3),
            ResidualLayer(128, 3),
            ResidualLayer(128, 3),
            ResidualLayer(128, 3)
        )
        self.DeconvBlock = nn.Sequential(
            DeconvLayer(128, 64, 3, 2, 1),
            nn.ReLU(),
            DeconvLayer(64, 32, 3, 2, 1),
            nn.ReLU(),
            ConvLayer(32, 3, 9, 1, norm="None")
        )

    def forward(self, x):
        x = self.ConvBlock(x)
        x = self.ResidualBlock(x)
        out = self.DeconvBlock(x)
        return out


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, norm="instance"):
        super(ConvLayer, self).__init__()
        # Padding Layers
        padding_size = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(padding_size)

        # Convolution Layer
        self.conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

        # Normalization Layers
        self.norm_type = norm
        if norm == "instance":
            self.norm_layer = nn.InstanceNorm2d(out_channels, affine=True)
        elif norm == "batch":
            self.norm_layer = nn.BatchNorm2d(out_channels, affine=True)

    def forward(self, x):
        x = self.reflection_pad(x)
        x = self.conv_layer(x)
        if self.norm_type == "None":
            out = x
        else:
            out = self.norm_layer(x)
        return out


class ResidualLayer(nn.Module):
    """
    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385
    """

    def __init__(self, channels=128, kernel_size=3):
        super(ResidualLayer, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size, stride=1)
        self.relu = nn.ReLU()
        self.conv2 = ConvLayer(channels, channels, kernel_size, stride=1)

    def forward(self, x):
        identity = x  # preserve residual
        out = self.relu(self.conv1(x))  # 1st conv layer + activation
        out = self.conv2(out)  # 2nd conv layer
        out = out + identity  # add residual
        return out


class DeconvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, output_padding, norm="instance"):
        super(DeconvLayer, self).__init__()

        # Transposed Convolution
        padding_size = kernel_size // 2
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding_size,
                                                 output_padding)

        # Normalization Layers
        self.norm_type = norm
        if norm == "instance":
            self.norm_layer = nn.InstanceNorm2d(out_channels, affine=True)
        elif norm == "batch":
            self.norm_layer = nn.BatchNorm2d(out_channels, affine=True)

    def forward(self, x):
        x = self.conv_transpose(x)
        if self.norm_type == "None":
            out = x
        else:
            out = self.norm_layer(x)
        return out


# ------------------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------------------
# VGG
class VGG16(nn.Module):
    def __init__(self, vgg_path="models/vgg16-00b39a1b.pth"):
        super(VGG16, self).__init__()
        # Load VGG Skeleton, Pretrained Weights
        vgg16_features = models.vgg16(pretrained=False)
        vgg16_features.load_state_dict(torch.load(vgg_path), strict=False)
        self.features = vgg16_features.features

        # Turn-off Gradient History
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        layers = {'3': 'relu1_2', '8': 'relu2_2', '15': 'relu3_3', '22': 'relu4_3'}
        features = {}
        for name, layer in self.features._modules.items():
            x = layer(x)
            if name in layers:
                features[layers[name]] = x
                if name == '22':
                    break

        return features


# ------------------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------------------
# Calculates Gram Matrix gram(f) = f * f^T
def gram(f):
    b, c, h, w = f.shape  # batch size, num channels, height, width
    f = f.view(b, c, h * w)
    # Normalize gram values to make gram matrices of differently-sized matrices comparable
    return torch.bmm(f, f.transpose(1, 2)) / (c * h * w)


# Convert tensor to image, where tensor contains BGR image data from [0.0, 255.0] in (C, H, W) format
# Returns image in BGR format
def to_image(tensor):
    img = tensor.cpu().detach().clone().squeeze().numpy()
    img = img.transpose(1, 2, 0)  # Transpose from [C, H, W] -> [H, W, C]
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# Show image (img must be BGR and from [0.0, 255.0])
def imshow(img, title=None):
    img = np.array(img / 255).clip(0, 1)  # imshow() only accepts float [0,1] or int [0,255]
    plt.imshow(img)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


# DELETE THIS!!!
# Preprocessing ~ Image to Tensor
def itot(img, max_size=None):
    # Rescale the image
    if max_size is None:
        itot_t = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255))
        ])
    else:
        H, W, C = img.shape
        image_size = tuple([int((float(max_size) / max([H, W])) * x) for x in [H, W]])
        itot_t = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255))
        ])

    # Convert image to tensor
    tensor = itot_t(img)

    # Add the batch_size dimension
    tensor = tensor.unsqueeze(dim=0)
    return tensor

# ------------------------------------------------------------------------------------------------------------------


def train():
    # Set random seeds for reproducibility
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # Load networks
    transformer = StyleTransfer().to(device)
    VGG = VGG16().to(device)

    # Get Style Features
    imagenet_neg_mean = torch.tensor([-103.939, -116.779, -123.68], dtype=torch.float32).reshape(1, 3, 1, 1).to(device)
    imagenet_pos_mean = torch.tensor([103.939, 116.779, 123.68], dtype=torch.float32).reshape(1, 3, 1, 1).to(device)
    content_image = cv2.imread(CONTENT_IMAGE_PATH)
    style_image = cv2.imread(STYLE_IMAGE_PATH)
    content_tensor = itot(content_image).to(device)
    style_tensor = itot(style_image).to(device)
    style_tensor = style_tensor.add(imagenet_neg_mean)
    b, c, h, w = style_tensor.shape
    style_features = VGG(style_tensor.expand([BATCH_SIZE, c, h, w]))
    style_gram = {}
    for key, value in style_features.items():
        style_gram[key] = gram(value)

    # Optimizer settings
    optimizer = optim.Adam(transformer.parameters(), lr=ADAM_LR)

    # Loss trackers
    batch_content_loss_sum = 0
    batch_style_loss_sum = 0
    batch_total_loss_sum = 0

    # Optimization/Training Loop
    # batch_count = 1
    for epoch in range(NUM_EPOCHS):
        print("========Epoch {}/{}========".format(epoch + 1, NUM_EPOCHS))
        content_batch = content_tensor
        # for content_batch, _ in train_loader:
        # Get current batch size in case of odd batch sizes
        curr_batch_size = content_batch.shape[0]

        # Free-up unneeded cuda memory
        torch.cuda.empty_cache()

        # Zero-out Gradients
        optimizer.zero_grad()

        # Generate images and get features
        content_batch = content_batch[:, [2, 1, 0]].to(device)
        generated_batch = transformer(content_batch)
        content_features = VGG(content_batch.add(imagenet_neg_mean))
        generated_features = VGG(generated_batch.add(imagenet_neg_mean))

        # Content Loss
        MSELoss = nn.MSELoss().to(device)
        content_loss = CONTENT_WEIGHT * MSELoss(generated_features['relu2_2'], content_features['relu2_2'])
        batch_content_loss_sum += content_loss

        # Style Loss
        style_loss = 0
        for key, value in generated_features.items():
            s_loss = MSELoss(gram(value), style_gram[key][:curr_batch_size])
            style_loss += s_loss
        style_loss *= STYLE_WEIGHT
        batch_style_loss_sum += style_loss.item()

        # Total Loss
        total_loss = content_loss + style_loss
        batch_total_loss_sum += total_loss.item()

        # Backprop and Weight Update
        total_loss.backward()
        optimizer.step()

        if epoch % 1 == 0:
            plt.close('all')
            fig = plt.figure(figsize=(7, 3))
            fig.add_subplot(1, 3, 1)
            imshow(to_image(content_tensor), title='Epoch {}: Content'.format(epoch + 1))
            fig.add_subplot(1, 3, 2)
            imshow(to_image(style_tensor.add(imagenet_pos_mean)), title='Epoch {}: Style'.format(epoch + 1))
            fig.add_subplot(1, 3, 3)
            imshow(to_image(generated_batch), title='Epoch {}: Transformed'.format(epoch + 1))
        print("\tLoss:\t{:.2f}".format(batch_total_loss_sum))


if __name__ == '__main__':
    train()
