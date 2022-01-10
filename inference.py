import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import random
import torchvision.transforms as transforms

from cnn import StyleTransfer
from classifier import ArtistClassifier


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
MODEL_FILENAME = None

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

print('Preparing networks!')
transform = transforms.Compose([
    transforms.ToTensor(), transforms.CenterCrop(256),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

classifier = ArtistClassifier(state_dict_filename='models/best-2.pth', num_classes=19, device=device)
classifier.eval()
if MODEL_FILENAME is None or not os.listdir(model_dir).__contains__(MODEL_FILENAME):
    raise NotImplementedError(MODEL_FILENAME)
net = StyleTransfer(state_dict_filename=model_dir + MODEL_FILENAME, device=device)
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
            # If image is a weird size, ignore it
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
