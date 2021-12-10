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

# Acknowledgements:
# Much of this code (dealing with the pre-trained VGG and crafting the content and first style loss functions via the
# VGG is adapted from this tutorial: https://pytorch.org/tutorials/advanced/neural_style_tutorial.html

# ------------------------------------------------------------------------------------------------------------------
# HYPERPARAMETERS
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# imsize = 512 if torch.cuda.is_available() else 128  # use small size if no gpu

# desired depth layers to compute content/style losses :
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']


# ------------------------------------------------------------------------------------------------------------------


class ContentLoss(nn.Module):
    def __init__(self, target, ):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL
    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)


class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


# create a module to normalize input image so we can easily put it in a nn.Sequential
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = mean.clone().detach().view(-1, 1, 1)
        self.std = std.clone().detach().view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


# Returns modified pre-trained VGG model to include content and style loss layers after the specified conv layers
# Also returns list of content and style loss layers to iterate through
def get_VGG_with_losses(content_image, style_image,
                        content_layers=content_layers_default, style_layers=style_layers_default):
    ######################################################################
    # VGG networks are trained on images with each channel
    # normalized by mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225].
    # We will use them to normalize the image before sending it into the network.
    normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    # Lists of loss layers
    content_losses = []
    style_losses = []

    # build model sequentially from pre-trained VGG
    vgg = models.vgg19(pretrained=True).features.to(device).eval()
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in vgg.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ContentLoss and StyleLoss layers we insert below.
            # So we replace with out-of-place ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_image).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_image).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break
    model = model[:i + 1]

    return model, content_losses, style_losses


def main():
    # artist_dataset = get_dataset(use_resized=True, save_pickle=False, load_pickle=True, wordy=True)
    # _, input_height, input_width = artist_dataset[0][0].numpy().shape

    # In order to get content_losses and style_losses, we must call get_VGG_with_losses() with the content and style
    # image for this training iteration. Best to do multiple steps with same content and style images for efficiency
    # If it seems too inefficient, maybe doing things this way would be better:
    # https://github.com/leongatys/PytorchNeuralStyleTransfer/blob/master/NeuralStyleTransfer.ipynb ?

    class StyleTransfer(nn.Module):
        def __init__(self):
            super(StyleTransfer, self).__init__()

            # Dimensions after conv2d: x_out = x_in - kernel_size + 1
            # Dimensions after maxpool2d: x_out = floor[(x_in - kernel_size) / stride + 1]
            # Dimensions after convtranspose2d: x_out = x_in + kernel_size - 1
            # Dimensions after maxunpool2d: x_out = stride * (x_in - 1) + kernel_size
            self.relu = nn.ReLU()

            self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=(8, 8))
            self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2, return_indices=True)
            self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3))
            self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2, return_indices=True)
            self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3))

            self.unconv1 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=(3, 3))
            self.unpool1 = nn.MaxUnpool2d(kernel_size=(2, 2), stride=2)
            self.unconv2 = nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=(3, 3))
            self.unpool2 = nn.MaxUnpool2d(kernel_size=(2, 2), stride=2)
            self.unconv3 = nn.ConvTranspose2d(in_channels=8, out_channels=3, kernel_size=(8, 8))

        def forward(self, x):
            out1 = self.conv1(x)
            out1 = self.relu(out1)
            out1_pooled, indices1 = self.pool1(out1)
            out2 = self.conv2(out1_pooled)
            out2 = self.relu(out2)
            out2_pooled, indices2 = self.pool2(out2)
            out3 = self.conv3(out2_pooled)
            out3 = self.relu(out3)

            out4 = self.unconv1(out3)
            out4 = self.relu(out4)
            out4_unpooled = self.unpool1(out4, indices2, output_size=out2.size())
            out5 = self.unconv2(out4_unpooled)
            out5 = self.relu(out5)
            out5_unpooled = self.unpool2(out5, indices1, output_size=out1.size())
            out6 = self.unconv3(out5_unpooled)
            return self.relu(out6)

        # Returns number of trainable parameters in a network
        def count_parameters(self):
            return sum(p.numel() for p in self.parameters() if p.requires_grad)

    model = StyleTransfer()
    model.requires_grad_(True)
    optimizer = optim.Adam(model.parameters(), lr=0.05)

    # desired size of the output image
    imsize = 512 if torch.cuda.is_available() else 256  # use small size if no gpu

    loader = transforms.Compose([
        transforms.Resize(imsize),  # scale imported image
        transforms.ToTensor()])  # transform it into a torch tensor

    def image_loader(image_name):
        image = Image.open(image_name)
        # fake batch dimension required to fit network's input dimensions
        image = loader(image).unsqueeze(0)
        with torch.no_grad():
            image.clamp_(0, 1)
        return image.to(device, torch.float)

    style_img = image_loader("picasso.jpg")
    content_img = image_loader("dancing.jpg")

    unloader = transforms.ToPILImage()

    def imshow(tensor, title=None):
        image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
        image = image.squeeze(0)  # remove the fake batch dimension
        image = unloader(image)
        plt.imshow(image)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)  # pause a bit so that plots are updated

    # TRAINING LOOP
    vgg, content_losses, style_losses = get_VGG_with_losses(content_image=content_img, style_image=style_img)
    vgg.requires_grad_(True)
    content_weight = 1
    style_weight = 1000
    prev_loss = 0.0
    prev_transformed = 0
    for i in range(1, 301):
        optimizer.zero_grad()
        transformed = model(content_img)
        transformed.requires_grad_(True)
        vgg(transformed)

        loss = torch.tensor([0.0])
        for content_loss_layer in content_losses:
            loss += content_weight * content_loss_layer.loss
        for style_loss_layer in style_losses:
            loss += style_weight * style_loss_layer.loss
        loss.backward()
        optimizer.step()

        if abs(loss.item() - prev_loss) < 0.0001:
            print(np.sum(np.abs((prev_transformed - transformed).detach().numpy()[0])))

        prev_transformed = transformed
        prev_loss = loss.item()

        if i % 20 == 0 or i <= 10 or loss.item() > 40:
            print('Epoch: {},    loss: {}'.format(i, loss.item()))
        if i % 100 == 0 or i == 1:
            fig = plt.figure(figsize=(7, 3))
            fig.add_subplot(1, 3, 1)
            imshow(content_img, title='Epoch {}: Content'.format(i))
            fig.add_subplot(1, 3, 2)
            imshow(style_img, title='Epoch {}: Style'.format(i))
            fig.add_subplot(1, 3, 3)
            imshow(transformed, title='Epoch {}: Transformed'.format(i))


if __name__ == '__main__':
    main()
