import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.models as models
import random
import torchvision.transforms as transforms

from transfer_network import StyleTransfer


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, size=None):
        super(AdaptiveConcatPool2d, self).__init__()
        sz = size or 1
        self.ap = nn.AdaptiveAvgPool2d(sz)
        self.mp = nn.AdaptiveMaxPool2d(sz)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)


class Normalize:
    def __init__(self, mean, std, inplace=False, dtype=torch.float):
        self.mean = torch.as_tensor(mean, dtype=dtype)[None, :, None, None]
        self.std = torch.as_tensor(std, dtype=dtype)[None, :, None, None]
        self.inplace = inplace

    def __call__(self, tensor):
        if not self.inplace:
            tensor = tensor.clone()

        tensor.sub_(self.mean).div_(self.std)
        return tensor


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

artists = ['Alfred_Sisley', 'Amedeo_Modigliani', 'Andy_Warhol', 'Edgar_Degas',
           'Francisco_Goya', 'Henri_Matisse', 'Leonardo_da_Vinci', 'Marc_Chagall',
           'Mikhail_Vrubel', 'Pablo_Picasso', 'Paul_Gauguin', 'Paul_Klee',
           'Peter_Paul_Rubens', 'Pierre-Auguste_Renoir', 'Rembrandt', 'Rene_Magritte',
           'Sandro_Botticelli', 'Titian', 'Vincent_van_Gogh']

# -------------------------------------------------------------------------------------------------------------------
# GENERAL PARAMETERS
# -------------------------------------------------------------------------------------------------------------------
STYLE_METHOD = 'random'
ARTIST = 'Pablo_Picasso'

model_dir = 'models/' + ARTIST + '/' + STYLE_METHOD + '/'
MANUAL_MODEL_FILENAME = None  # If this is None, automatically finds model filename with horrible if-else

DISPLAY = True  # If DISPLAY, transform CONTENT_IMG and display results; if not, evaluate model outputs with classifier

# -------------------------------------------------------------------------------------------------------------------
# PARAMETERS FOR DISPLAYING AND MAKING FIGURES
# -------------------------------------------------------------------------------------------------------------------
CONTENT_IMG = 'cuteimages/' + 'landscape.jpg'  # Image to transform if DISPLAY == TRUE
CONTENT_SIZE_W = 1024  # Resize content image to have width of CONTENT_SIZE_W if CONTENT_SIZE_W > 0, else don't resize

SHARPEN = False
BLUR = False
if SHARPEN:
    sharpen_val = 50
if BLUR:
    blur_sigma = 1.0
    blur_kernel = (3, 3)

# -------------------------------------------------------------------------------------------------------------------
# PARAMETERS FOR QUANTITATIVE EVALUATION VIA CLASSIFIER
# -------------------------------------------------------------------------------------------------------------------
CONTENT_DIR = 'images/content/'  # Directory to pull random content images to evaluate classifier with from
RESIZE_IMGS = True  # Resize images to (RESIZE_SIZE, RESIZE_SIZE) during quantitative evaluation via classifier
RESIZE_SIZE = 1024
NUM_IMAGES = 133  # Number of random content images to grab and evaluate classifier with
# -------------------------------------------------------------------------------------------------------------------

# Horrible coding practices to help automate the process of calling this function
model_filename = MANUAL_MODEL_FILENAME
if MANUAL_MODEL_FILENAME is None:
    if ARTIST == 'Pablo_Picasso':
        if STYLE_METHOD == 'average':
            model_filename = 'transfer2_17-40_5.pth'
        elif STYLE_METHOD == 'classifier':
            model_filename = 'transfer2_17-1500000_33.pth'
        elif STYLE_METHOD == 'cycle':
            model_filename = 'transfer_17-35_24.pth'
        elif STYLE_METHOD == 'random':
            model_filename = 'Pablo_Picasso_transfer2_280.pth'
        elif STYLE_METHOD == 'smartaverage':
            print('invalid jawn')
            exit()
        else:
            print('invalid jawn')
            exit()
    elif ARTIST == 'Leonardo_da_Vinci':
        if STYLE_METHOD == 'average':
            print('invalid jawn')
            exit()
        elif STYLE_METHOD == 'classifier':
            model_filename = 'transfer_17-2000000_20.pth'
        elif STYLE_METHOD == 'cycle':
            model_filename = 'transfer_17-40_33.pth'
        elif STYLE_METHOD == 'random':
            model_filename = 'transfer_17-40_32.pth'
        elif STYLE_METHOD == 'smartaverage':
            model_filename = 'transfer_17-45_3.pth'
        else:
            print('invalid jawn')
            exit()
    elif ARTIST == 'Vincent_van_Gogh':
        if STYLE_METHOD == 'average':
            print('invalid jawn')
            exit()
        elif STYLE_METHOD == 'classifier':
            model_filename = 'transfer_17-2000000_15.pth'
        elif STYLE_METHOD == 'cycle':
            model_filename = 'transfer_17-40_30.pth'
        elif STYLE_METHOD == 'random':
            model_filename = 'transfer2_17-40_30.pth'
        elif STYLE_METHOD == 'smartaverage':
            model_filename = 'transfer_17-45_10.pth'
        else:
            print('invalid jawn')
            exit()
    elif ARTIST == 'Rembrandt':
        if STYLE_METHOD == 'average':
            print('invalid jawn')
            exit()
        elif STYLE_METHOD == 'classifier':
            model_filename = 'transfer_17-2000000_15.pth'
        elif STYLE_METHOD == 'cycle':
            model_filename = 'transfer_17-40_33.pth'
        elif STYLE_METHOD == 'random':
            model_filename = 'transfer_17-40_33.pth'
        elif STYLE_METHOD == 'smartaverage':
            print('invalid jawn')
            exit()
        else:
            print('invalid jawn')
            exit()
    else:
        print('invalid jawn')
        exit()

print('Preparing networks!')

classifier = models.resnet50(pretrained=False)
# Remove final layers and append the correct outputs
modules = list(classifier.children())
modules.pop(-1)
modules.pop(-1)
feature_layers = nn.Sequential(nn.Sequential(*modules))
feature_children = list(feature_layers.children())
# Append the layers we need (19 classes in this classifier)
feature_children.append(nn.Sequential(
    AdaptiveConcatPool2d(), Flatten(), nn.BatchNorm1d(4096), nn.Dropout(p=0.375), nn.Linear(4096, 512),
    nn.ReLU(), nn.BatchNorm1d(512), nn.Dropout(p=0.75), nn.Linear(512, 19)
))
classifier = nn.Sequential(*feature_children)
sd = torch.load('models/best-2.pth', map_location=device)
classifier.load_state_dict(sd['model'], strict=True)
for param in classifier.parameters():
    param.requires_grad = False
classifier = classifier.double().to(device)
classifier.eval()

transform = transforms.Compose([
    transforms.ToTensor(), transforms.CenterCrop(256),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

net = StyleTransfer().double()
net.load_state_dict(torch.load(model_dir + model_filename, map_location=device),
                    strict=True)  # 'transfer_17-35_24.pth'
net = net.double().to(device)
net.eval()

index = artists.index(ARTIST)

if DISPLAY:
    im = cv2.imread(CONTENT_IMG)
    if CONTENT_SIZE_W > 0:
        h, w, c = im.shape
        content_images = [cv2.resize(im, (CONTENT_SIZE_W, int(h * CONTENT_SIZE_W / w)))]
    else:
        content_images = [im]
    if STYLE_METHOD == 'random' or STYLE_METHOD == 'artist' or STYLE_METHOD == 'average':
        style_image = cv2.imread(model_dir + 'style.jpg')
else:
    content_images = []
    content_files = os.listdir(CONTENT_DIR)
    random.shuffle(content_files)

    for file in content_files:
        if not file.__contains__('.jpg') and not file.__contains__('.JPEG'):
            continue
        im = cv2.imread(CONTENT_DIR + file)
        if im is None:
            continue
        if RESIZE_IMGS:
            content_images.append(cv2.resize(im, (RESIZE_SIZE, RESIZE_SIZE)))
        else:
            # If image is too big, resize so large dimension is 1024
            h, w, c = im.shape
            if h > 1600 or w > 1024 or h < 224 or w < 224:
                continue
            else:
                content_images.append(im)
        if len(content_images) == NUM_IMAGES:
            break

    print('Grabbed {} images!\n'.format(len(content_images)))

print('Running model(s)! :)')
num_correct = 0
total = 0
for i in range(len(content_images)):
    with torch.no_grad():
        input_img = content_images[i]
        input_tensor = torch.from_numpy(input_img.transpose(2, 0, 1)).double().unsqueeze(0).to(device)

        if DISPLAY:
            content_img = cv2.cvtColor(input_img.astype('uint8'), cv2.COLOR_BGR2RGB).astype('uint8')

        out_tensor = net(input_tensor)
        out_img = out_tensor.cpu().squeeze().numpy()[[2, 1, 0]].transpose(1, 2, 0).clip(0, 255).astype('uint8')

        if DISPLAY:
            if BLUR:
                out_img = cv2.GaussianBlur(out_img, ksize=blur_kernel, sigmaX=blur_sigma, sigmaY=blur_sigma).\
                    astype('uint8')
            if SHARPEN:
                out_img = cv2.filter2D(out_img, -1, np.array([[-1, -1, -1],
                                                              [-1, sharpen_val, -1],
                                                              [-1, -1, -1]]) / (sharpen_val - 8))
            fig = plt.figure(figsize=(18, 5))
            if STYLE_METHOD != 'classifier' and STYLE_METHOD != 'smartaverage':
                fig.add_subplot(1, 3, 1)
            else:
                fig.add_subplot(1, 2, 1)
            plt.imshow(cv2.cvtColor(input_img.astype('uint8'), cv2.COLOR_BGR2RGB).astype('uint8'),
                       interpolation='nearest',
                       aspect='auto')
            plt.title('Content', fontsize=28)
            plt.pause(0.001)

            if STYLE_METHOD == 'average' or STYLE_METHOD == 'random':
                fig.add_subplot(1, 3, 2)
                plt.imshow(cv2.cvtColor(style_image.astype('uint8'), cv2.COLOR_BGR2RGB).astype('uint8'),
                           interpolation='nearest', aspect='auto')
                plt.title('Style', fontsize=28)
                plt.pause(0.001)

            if STYLE_METHOD != 'classifier' and STYLE_METHOD != 'smartaverage':
                fig.add_subplot(1, 3, 3)
            else:
                fig.add_subplot(1, 2, 2)
            plt.imshow(out_img, interpolation='nearest', aspect='auto')
            plt.title('Transformed', fontsize=28)
            plt.pause(0.001)
            plt.savefig('figs/' + ARTIST + '_' + STYLE_METHOD + '.png')
            plt.show()
        else:
            out_tensor_for_classifier = transform(out_img).unsqueeze(0).double().to(device)
            out_class = torch.argmax(torch.softmax(classifier(out_tensor_for_classifier).squeeze(), dim=0),
                                     dim=0).cpu().item()
            print('Pred={}\tActual={}\timage_num={}'.format(artists[out_class], artists[index], i + 1))
            if out_class == index:
                num_correct += 1
            total += 1
            del out_tensor_for_classifier
        del input_tensor
        del out_tensor

if not DISPLAY:
    print('Acc={}'.format(round(100 * num_correct / total, 2)))
