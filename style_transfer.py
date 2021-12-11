import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.models as models
import random
import time
from dataset import get_content_dataset, get_painting_dataset, get_avg_dataset

# Acknowledgements:
# Much of this code (residual transfer network, dealing with the pre-trained VGG and crafting the content and first
# style loss functions via the VGG) is adapted from and heavily inspired by the code from the github linked below:
# https://github.com/rrmina/fast-neural-style-pytorch

# ------------------------------------------------------------------------------------------------------------------
# HYPERPARAMETERS
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}

TRAIN_SIZE = 224 if torch.cuda.is_available() else 128  # use small size if no gpu

# Can be 'random', 'average', 'cycle', or 'classifier'
STYLE_METHOD = 'random'
ARTIST = 'Jackson_Pollock'  # 'Albrecht_Dürer'

NUM_EPOCHS = 200
BATCH_SIZE = 4 if torch.cuda.is_available() else 1
CONTENT_DATA_SIZE = 256
CONTENT_WEIGHT = 17  # 17
STYLE_WEIGHT = 50  # 25
LR = 0.0024
SEED = 0
SAVE_EVERY = 10  # How often to save model during training in epochs


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
            nn.ReLU(),
            ConvLayer(128, 128, 1, 1),  # ADDED CONV LAYER WITH 1 x 1 KERNEL
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
            DeconvLayer(128, 128, 1, 1, 0),  # ADDED CONV LAYER WITH 1 x 1 KERNEL
            nn.ReLU(),
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
        if kernel_size > 1:
            # Padding Layers
            padding_size = kernel_size // 2
            self.reflection_pad = nn.ReflectionPad2d(padding_size)
        else:
            self.reflection_pad = nn.Identity()

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
        vgg16 = models.vgg16(pretrained=False)
        vgg16.load_state_dict(torch.load(vgg_path), strict=False)
        self.features = vgg16.features

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
# Returns image in RGB format
def to_image(tensor):
    if len(tensor.size()) < 4:
        img = tensor.cpu().detach().clone().numpy()
    else:
        img = tensor.cpu().detach().clone().squeeze().numpy()
    img = img[[2, 1, 0]].transpose(1, 2, 0)  # Convert to RGB and transpose from [C, H, W] -> [H, W, C]
    return img


# Show image (img must be BGR and from [0.0, 255.0])
def imshow(img, title=None):
    img = np.array(img / 255).clip(0, 1)  # imshow() only accepts float [0,1] or int [0,255]
    plt.imshow(img)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


# ------------------------------------------------------------------------------------------------------------------

def train():
    # Set random seeds for reproducibility
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # Load networks
    transfer = StyleTransfer().double().to(device)
    VGG = VGG16().double().to(device)

    # Content dataset
    content_dataset = get_content_dataset(CONTENT_DATA_SIZE, rescale_height=TRAIN_SIZE, rescale_width=TRAIN_SIZE)
    content_loader = torch.utils.data.DataLoader(content_dataset, batch_size=BATCH_SIZE, shuffle=True, **kwargs)

    # Get Style Features
    imagenet_neg_mean = torch.tensor([-103.939, -116.779, -123.68], dtype=torch.float32).reshape(1, 3, 1, 1).to(device)
    imagenet_pos_mean = torch.tensor([103.939, 116.779, 123.68], dtype=torch.float32).reshape(1, 3, 1, 1).to(device)

    if STYLE_METHOD == 'random':
        style_dataset = get_painting_dataset(for_classifier=False, rescale_height=TRAIN_SIZE, rescale_width=TRAIN_SIZE,
                                             use_resized=True, save_pickle=False, load_pickle=True, wordy=True)
        style_tensor = style_dataset[ARTIST][random.randint(0, len(style_dataset[ARTIST]) - 1)] \
            .double().to(device).add(imagenet_neg_mean)
        b, c, h, w = style_tensor.shape
        style_features = VGG(style_tensor.expand([BATCH_SIZE, c, h, w]))
        style_gram = {}
        for key, value in style_features.items():
            style_gram[key] = gram(value)
    elif STYLE_METHOD == 'average':
        style_dataset = get_avg_dataset(rescale_height=TRAIN_SIZE, rescale_width=TRAIN_SIZE, wordy=True)
        style_tensor = style_dataset[ARTIST].double().to(device).add(imagenet_neg_mean)
        b, c, h, w = style_tensor.shape
        style_features = VGG(style_tensor.expand([BATCH_SIZE, c, h, w]))
        style_gram = {}
        for key, value in style_features.items():
            style_gram[key] = gram(value)
    elif STYLE_METHOD == 'cycle':
        style_dataset = get_painting_dataset(for_classifier=False, rescale_height=TRAIN_SIZE, rescale_width=TRAIN_SIZE,
                                             use_resized=True, save_pickle=False, load_pickle=True, wordy=True)
        style_gram_cycle = []
        for t in style_dataset[ARTIST]:
            style_tensor = t.double().to(device).add(imagenet_neg_mean)
            b, c, h, w = style_tensor.shape
            style_features = VGG(style_tensor.expand([BATCH_SIZE, c, h, w]))
            style_gram = {}
            for key, value in style_features.items():
                style_gram[key] = gram(value)
            style_gram_cycle.append(style_gram)

    # Optimizer settings
    optimizer = optim.Adam(transfer.parameters(), lr=LR, weight_decay=0.0001)
    MSELoss = nn.MSELoss().to(device)

    # Optimization/Training Loop
    batch_count = 0
    save_dir_prefix = 'models/' + ARTIST + '/' + STYLE_METHOD + '/' + 'transfer_' + str(CONTENT_WEIGHT) + '-' + \
                      str(STYLE_WEIGHT)
    print('Training!')
    start = time.time()
    for epoch in range(NUM_EPOCHS):
        print("========Epoch {}/{}========".format(epoch + 1, NUM_EPOCHS))
        for content_batch, _ in content_loader:
            # Loss trackers over each batch
            batch_content_loss_sum = torch.tensor(0, dtype=torch.float32, device=device)
            batch_style_loss_sum = torch.tensor(0, dtype=torch.float32, device=device)
            batch_total_loss_sum = torch.tensor(0, dtype=torch.float32, device=device)

            # Free-up unneeded cuda memory
            torch.cuda.empty_cache()

            # Zero-out Gradients
            optimizer.zero_grad()

            # Generate images and get features
            content_batch = content_batch.to(device)
            generated_batch = transfer(content_batch)
            content_features = VGG(content_batch.add(imagenet_neg_mean))
            generated_features = VGG(generated_batch.add(imagenet_neg_mean))

            # Content Loss and Style Loss
            content_loss = MSELoss(generated_features['relu2_2'], content_features['relu2_2'])
            content_loss *= CONTENT_WEIGHT
            batch_content_loss_sum += content_loss

            if STYLE_METHOD == 'cycle':
                index = batch_count % len(style_dataset[ARTIST])
                # style_tensor = style_dataset[ARTIST][index].double().to(device).add(imagenet_neg_mean)
                style_gram = style_gram_cycle[index]
            style_loss = 0
            for key, value in generated_features.items():
                s_loss = MSELoss(gram(value), style_gram[key])
                style_loss += s_loss
            style_loss *= STYLE_WEIGHT
            batch_style_loss_sum += style_loss

            # Total Loss
            total_loss = content_loss + style_loss
            batch_total_loss_sum += total_loss

            # Backprop and Weight Update
            total_loss.backward()
            optimizer.step()

            if batch_count % 1 == 0:
                plt.close('all')
                fig = plt.figure(figsize=(7, 3))
                fig.add_subplot(1, 3, 1)
                imshow(to_image(content_batch[0]), title='Epoch {}: Content'.format(epoch + 1))
                fig.add_subplot(1, 3, 2)
                imshow(to_image(style_tensor.add(imagenet_pos_mean)), title='Epoch {}: Style'.format(epoch + 1))
                fig.add_subplot(1, 3, 3)
                imshow(to_image(generated_batch[0]), title='Epoch {}: Transformed'.format(epoch + 1))
                plt.show()

            batch_count += 1

            # print("\tContent Loss:\t{:.2f}".format(batch_content_loss_sum.item()))
            # print("\tStyle Loss:\t{:.2f}".format(batch_style_loss_sum.item()))
            # print("\tTotal Loss:\t{:.2f}\n".format(batch_total_loss_sum.item()))

            # Clean created tensors
            del batch_content_loss_sum
            del batch_style_loss_sum
            del batch_total_loss_sum
            del content_batch
            del generated_batch
            del content_features
            del generated_features
            if STYLE_METHOD == 'cycle':
                del style_gram
            del content_loss
            del style_loss
            del total_loss

        if epoch % SAVE_EVERY == 0:
            torch.save(transfer.state_dict(), save_dir_prefix + '_' + str(epoch) + '.pth')

    print('\n\nTRAINED IN {:.2f} SEC'.format(time.time() - start))
    transfer.eval()
    transfer.cpu()
    torch.save(transfer.state_dict(), save_dir_prefix + '_final.pth')


if __name__ == '__main__':
    train()
