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

# Acknowledgements:
# The part of this code dealing with the pre-trained VGG and crafting the content and first style loss functions
# via the VGG) is adapted from and heavily inspired by {Johnson, Fei Fei paper} and the code from the
# github linked below:
# https://github.com/rrmina/fast-neural-style-pytorch

# ------------------------------------------------------------------------------------------------------------------
# HYPERPARAMETERS
# ------------------------------------------------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}

TRAIN_SIZE = 224 if torch.cuda.is_available() else 128  # use small size if no gpu
BATCH_INFO_EVERY = 12  # How often to print/show during training, measured in batches

# The below parameters will be overwritten by the parameters train() is called with
STYLE_METHOD = 'random'  # Can be 'random', 'average', 'cycle', or 'classifier'
ARTIST = 'Albrecht_Dürer'  # 'Albrecht_Dürer'

NUM_EPOCHS = 200  # Number of epochs to train for
BATCH_SIZE = 4 if torch.cuda.is_available() else 1  # Number of images in each batch
CONTENT_DATA_SIZE = 256  # Number of ImageNet images to draw batches from
LR = 0.0024  # Adam learning rate

CONTENT_WEIGHT = 17  # 17
STYLE_WEIGHT = 25  # 25

SAVE_EVERY = 10  # How often to save model during training, measured in epochs
SEED = 2  # Random seed for RNGs, used for reproducibility


# ------------------------------------------------------------------------------------------------------------------
# VGG implementation that returns a dict of responses after predetermined conv2D layers instead of the final output
# ------------------------------------------------------------------------------------------------------------------
class VGG16(nn.Module):
    def __init__(self, just_content=False, vgg_path="models/vgg16-00b39a1b.pth"):
        super(VGG16, self).__init__()
        # Load VGG Skeleton, Pretrained Weights
        vgg16 = models.vgg16(pretrained=False)
        vgg16.load_state_dict(torch.load(vgg_path), strict=False)
        self.features = vgg16.features
        self.just_content = just_content

        # Turn-off Gradient History
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        if self.just_content:
            for name, layer in self.features._modules.items():
                x = layer(x)
                if name == 'relu2_2':
                    return x
        else:
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
# USEFUL METHODS
# ------------------------------------------------------------------------------------------------------------------
# Calculates Gram Matrix gram(f) = f^-T * f of a batch of 2-D matrices
# Can be thought of as the statistical overlap, and is often used in CNNs to capture texture information of responses
def gram(f):
    b, c, h, w = f.shape  # batch size, num channels, height, width
    f = f.view(b, c, h * w)
    # Normalize gram values to make gram matrices of differently-sized matrices comparable
    return torch.bmm(f, f.transpose(1, 2)) / (c * h * w)


# Save tensor as a .jpg image file
def save_tensor_image(filename, tensor):
    image = cv2.cvtColor(to_image(tensor).clip(0, 255).astype('uint8'), cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, image)


# Convert tensor to image, where tensor contains BGR image data from [0.0, 255.0] in (C, H, W) format
# Returns image in (H, W, C) RGB format with pixels from [0.0, 255.0]
def to_image(tensor):
    if len(tensor.size()) < 4:
        img = tensor.cpu().detach().clone().numpy()
    else:
        img = tensor.cpu().detach().clone().squeeze().numpy()
    img = img[[2, 1, 0]].transpose(1, 2, 0)  # Convert to RGB and transpose from [C, H, W] -> [H, W, C]
    return img


# Show image (img must be RGB and from [0.0, 255.0])
def imshow(img, title=None):
    # imshow() only accepts pixels [0.0, 1.0] or [0, 255], so transform to [0.0, 1.0]
    img = np.array(img / 255).clip(0, 1)
    plt.imshow(img)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


# ------------------------------------------------------------------------------------------------------------------
# TRAINS STYLE NETWORK FOR A GIVEN ARTIST using global parameters if none are specified
# ------------------------------------------------------------------------------------------------------------------
# Calculates total loss as weighted sum of content_loss and style_loss, where style_loss is calculated using the method
# specified by 'style_method'
# Trains with 'content_data_size / batch_size' batches per epoch for 'num_epochs' epochs, saves state dict every
# 'save_every' epochs
def train(style_method=STYLE_METHOD, artist=ARTIST, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE,
          content_data_size=CONTENT_DATA_SIZE, seed=SEED,
          content_weight=CONTENT_WEIGHT, style_weight=STYLE_WEIGHT, lr=LR, save_every=SAVE_EVERY):
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Load networks
    transfer = StyleTransfer().double().to(device)
    if style_method == 'classifier':
        VGG = VGG16(just_content=True).double().to(device)
        # classifier =
    else:
        VGG = VGG16().double().to(device)

    imagenet_neg_mean = torch.tensor([-103.939, -116.779, -123.68], dtype=torch.float32).reshape(1, 3, 1, 1).to(device)

    # Content dataset
    print('Getting content dataset!')
    content_dataset = get_content_dataset(content_data_size, rescale_height=TRAIN_SIZE, rescale_width=TRAIN_SIZE)
    content_loader = torch.utils.data.DataLoader(content_dataset, batch_size=batch_size, shuffle=True, **kwargs)

    # Get Style Features
    if len(os.listdir('models/' + artist + '/' + style_method + '/')) == 0:
        save_dir_prefix = 'models/' + artist + '/' + style_method + '/' + 'transfer_' + str(content_weight) + '-' + \
                          str(style_weight)
    else:
        save_dir_prefix = 'models/' + artist + '/' + style_method + '/' + 'transfer2_' + str(content_weight) + '-' + \
                          str(style_weight)

    print('Getting style dataset and features!')
    if style_method == 'random':
        style_dataset = get_painting_dataset(for_classifier=False, rescale_height=TRAIN_SIZE, rescale_width=TRAIN_SIZE,
                                             use_resized=True, save_pickle=False, load_pickle=True, wordy=False)
        style_tensor = style_dataset[artist][random.randint(0, len(style_dataset[artist]) - 1)] \
            .double().to(device).add(imagenet_neg_mean)
        b, c, h, w = style_tensor.shape
        style_features = VGG(style_tensor.expand([batch_size, c, h, w]))
        style_gram = {}
        for key, value in style_features.items():
            style_gram[key] = gram(value)
        if len(os.listdir('models/' + artist + '/' + style_method + '/')) == 0:
            save_tensor_image('models/' + artist + '/' + style_method + '/style.jpg', style_tensor)
        else:
            save_tensor_image('models/' + artist + '/' + style_method + '/style2.jpg', style_tensor)
    elif style_method == 'average':
        style_dataset = get_avg_dataset(rescale_height=TRAIN_SIZE, rescale_width=TRAIN_SIZE, wordy=False)
        style_tensor = style_dataset[artist].double().to(device).add(imagenet_neg_mean)
        b, c, h, w = style_tensor.shape
        style_features = VGG(style_tensor.expand([batch_size, c, h, w]))
        style_gram = {}
        for key, value in style_features.items():
            style_gram[key] = gram(value)
        save_tensor_image('models/' + artist + '/' + style_method + '/style.jpg', style_tensor)
    elif style_method == 'cycle':
        style_dataset = get_painting_dataset(for_classifier=False, rescale_height=TRAIN_SIZE, rescale_width=TRAIN_SIZE,
                                             use_resized=True, save_pickle=False, load_pickle=True, wordy=False)
        style_gram_cycle = []
        for t in style_dataset[artist]:
            style_tensor = t.double().to(device).add(imagenet_neg_mean)
            b, c, h, w = style_tensor.shape
            style_features = VGG(style_tensor.expand([batch_size, c, h, w]))
            style_gram = {}
            for key, value in style_features.items():
                style_gram[key] = gram(value)
            style_gram_cycle.append(style_gram)

    # Optimizer and loss function settings
    optimizer = optim.Adam(transfer.parameters(), lr=lr, weight_decay=0.0001)
    MSELoss = nn.MSELoss().to(device)

    # Optimization/Training Loop
    if style_method == 'random':
        method = 0
    elif style_method == 'average':
        method = 1
    elif style_method == 'cycle':
        method = 2
    elif style_method == 'classifier':
        method = 3
    else:
        print('enter valid style method!')
        return 0
    batch_count = 0
    print('Training!')
    start = time.time()
    epoch_start = start
    for epoch in range(num_epochs):
        print("========Epoch {}/{}========\tprev took {} secs".format(epoch + 1, num_epochs,
                                                                      round(time.time() - epoch_start, 2)))
        epoch_start = time.time()
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
            if method == 3:
                content_loss = MSELoss(generated_features, content_features)
            else:
                content_loss = MSELoss(generated_features['relu2_2'], content_features['relu2_2'])
            content_loss *= content_weight
            batch_content_loss_sum += content_loss

            if method == 3:
                print('u forgot to finihs hte classifier stuff mate')
            else:
                if method == 2:
                    index = batch_count % len(style_dataset[artist])
                    style_gram = style_gram_cycle[index]
                    if batch_count % BATCH_INFO_EVERY == 0:
                        style_tensor = style_dataset[artist][index].double()
                style_loss = 0
                for key, value in generated_features.items():
                    s_loss = MSELoss(gram(value), style_gram[key])
                    style_loss += s_loss
            style_loss *= style_weight
            batch_style_loss_sum += style_loss

            # Total Loss
            total_loss = content_loss + style_loss
            batch_total_loss_sum += total_loss

            # Backprop and Weight Update
            total_loss.backward()
            optimizer.step()

            # Print and show images every BATCH_INFO_EVERY batches
            if batch_count % BATCH_INFO_EVERY == 0:
                plt.close('all')
                fig = plt.figure(figsize=(7, 3))
                fig.add_subplot(1, 3, 1)
                imshow(to_image(content_batch[0]), title='Epoch {}: Content'.format(epoch + 1))
                fig.add_subplot(1, 3, 2)
                imshow(to_image(style_tensor), title='Epoch {}: Style'.format(epoch + 1))
                fig.add_subplot(1, 3, 3)
                imshow(to_image(generated_batch[0]), title='Epoch {}: Transformed'.format(epoch + 1))
                print("\tContent Loss:\t{:.2f}".format(batch_content_loss_sum.item()))
                print("\tStyle Loss:\t{:.2f}".format(batch_style_loss_sum.item()))
                print("\tTotal Loss:\t{:.2f}\n".format(batch_total_loss_sum.item()))

            batch_count += 1

            # Clean created tensors
            del batch_content_loss_sum
            del batch_style_loss_sum
            del batch_total_loss_sum
            del content_batch
            del generated_batch
            del content_features
            del generated_features
            if method == 2:
                del style_gram
            del content_loss
            del style_loss
            del total_loss

        if epoch % save_every == 0:
            torch.save(transfer.state_dict(), save_dir_prefix + '_' + str(epoch) + '.pth')

    print('\n\nTRAINED IN {:.2f} SEC'.format(time.time() - start))
    transfer.eval()
    transfer.cpu()
    torch.save(transfer.state_dict(), save_dir_prefix + '_' + str(num_epochs) + '.pth')


if __name__ == '__main__':

    train(style_method='cycle', artist='Jackson_Pollock', num_epochs=33, batch_size=4, content_data_size=256, seed=56,
          content_weight=17, style_weight=40, lr=0.0036, save_every=8)
    train(style_method='cycle', artist='Pablo_Picasso', num_epochs=33, batch_size=4, content_data_size=256, seed=5,
          content_weight=17, style_weight=40, lr=0.0036, save_every=8)
    train(style_method='random', artist='Georges_Seurat', num_epochs=25, batch_size=4, content_data_size=256, seed=3,
          content_weight=17, style_weight=50, lr=0.0048, save_every=8)
    train(style_method='random', artist='Claude_Monet', num_epochs=25, batch_size=4, content_data_size=256, seed=4,
          content_weight=17, style_weight=50, lr=0.0048, save_every=8)
