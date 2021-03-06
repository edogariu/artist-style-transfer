import numpy as np
import cv2
import pandas
import pickle
import torch
import os
import torchvision.transforms as transforms
import random

NUM_DICTS = 6  # Number of dictionaries to split dataset into when pickling (too big for one file)
DICT_SAVE_DIR = 'dicts/'
ARCHIVE_DIR = 'images/archive/'
CONTENT_DIR = 'images/content/'


def get_rescale_dims(dataset, total_paintings, rescale_width=-1, rescale_height=-1):
    # Rescale to avg dimensions if unspecified
    if rescale_width <= 0 or rescale_height <= 0:
        avg_height = 0.0
        avg_width = 0.0
        for images in dataset.values():
            for image in images:
                avg_height += image.shape[0] / total_paintings
                avg_width += image.shape[1] / total_paintings
        avg_height = int(avg_height)
        avg_width = int(avg_width)
        target_height = avg_height // 2
        target_width = avg_width // 2
    else:
        target_height = rescale_height
        target_width = rescale_width

    return target_height, target_width


def rescale(image, target_height, target_width):
    # Rescale each image to target dimensions

    # Source width and height in pixels
    src_height = image.shape[0]
    src_width = image.shape[1]

    # Scaling parameters
    h_s = target_height / src_height
    w_s = target_width / src_width

    # Affine matrices for width and height
    Affine_Mat_h = [0, h_s, target_height / 2 - h_s * src_height / 2]
    Affine_Mat_w = [w_s, 0, target_width / 2 - w_s * src_width / 2]
    M = np.c_[Affine_Mat_w, Affine_Mat_h].T

    return cv2.warpAffine(image, M, (target_width, target_height))


# Return average images for each artist
def get_avg_dataset(rescale_height=-1, rescale_width=-1, wordy=False):
    df = pandas.read_csv(ARCHIVE_DIR + 'artists.csv')

    # Replace spaces with _
    names = list(df['name'])
    for i in range(len(names)):
        names[i] = names[i].replace(' ', '_')

    # Get images for each artist
    total_paintings = np.sum(df['paintings'])

    dataset = {}
    for file in os.listdir(DICT_SAVE_DIR):
        if file.__contains__('full_int'):
            with open(DICT_SAVE_DIR + file, 'rb') as f:
                dataset.update(pickle.load(f))
    if wordy:
        print('Loaded!')

    target_height, target_width = get_rescale_dims(dataset, total_paintings,
                                                   rescale_height=rescale_height, rescale_width=rescale_width)

    avg_img = {}
    for artist in dataset.keys():
        avg_img[artist] = np.zeros((target_height, target_width, 3), dtype=float)
        for i in range(len(dataset[artist])):
            avg_img[artist] += rescale(dataset[artist][i], target_height=target_height,
                                       target_width=target_width) / len(dataset[artist])
        avg_img[artist] = torch.from_numpy(avg_img[artist].astype('uint8')
                                           .transpose((2, 0, 1))).view(3, target_height, target_width)

    return avg_img


def get_content_dataset(size, rescale_height, rescale_width):
    images = np.zeros((size, rescale_height, rescale_width, 3))
    count = 0
    content_files = os.listdir(CONTENT_DIR)
    random.shuffle(content_files)

    for file in content_files:
        im = cv2.imread(CONTENT_DIR + file)
        if im is None:
            continue

        images[count, :, :, :] = cv2.resize(im, (rescale_height, rescale_width)).astype(float)

        count += 1
        if count >= size:
            break
    images = images.transpose((0, 3, 1, 2))

    in_tensors = torch.from_numpy(images).view(size, 3, rescale_height, rescale_width)
    out_tensors = torch.from_numpy(np.zeros(size)).view(-1, 1)

    return torch.utils.data.TensorDataset(in_tensors, out_tensors)


# Script to load dataset of paintings (BGR, CxHxW, [0.0, 1.0]) with artist labels (from 0 to 49) if for_classifier==True
# If not, returns dictionary where artist is the key and the value is a list of all the paintings belonging to that
# artist as BGR tensors of shape (C x H x W) with values from [0, 255].
# Call with load_pickle=False to load from the dataset directly, and call with load_pickle=True
# to load from pre-saved dictionaries for faster loading
# Rescales paintings to (rescale_height x rescale_width) if specified, rescales to average dimensions // 2 if not
def get_painting_dataset(for_classifier=True, rescale_height=-1, rescale_width=-1, use_resized=True,
                         save_pickle=False, load_pickle=True, wordy=False):
    df = pandas.read_csv(ARCHIVE_DIR + 'artists.csv')

    # Replace spaces with _
    names = list(df['name'])
    for i in range(len(names)):
        names[i] = names[i].replace(' ', '_')

    # Get images for each artist
    dataset = {artist: [] for artist in names}

    count = 0
    count_failed = 0
    total_paintings = np.sum(df['paintings'])
    if not load_pickle:
        if use_resized:
            for i in range(len(dataset.keys())):
                num_paintings = df['paintings'][i]  # Number of paintings from each artist
                for num in range(1, num_paintings + 1):
                    im = cv2.imread(ARCHIVE_DIR + 'resized/resized/' + names[i] + '_' + str(num) + '.jpg')
                    if im is None:
                        count_failed += 1
                        continue
                    count += 1
                    if for_classifier:
                        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                        im = im.astype('float32') / 255
                    if count % 20 == 0 and wordy:
                        print(str(round(100 * count / (total_paintings - count_failed), 2)) + '%')
                    dataset[names[i]].append(np.array(im))
        else:
            for artist in names:
                dir = ARCHIVE_DIR + 'images/images/' + artist
                for file in os.listdir(dir):
                    im = cv2.imread(dir + '/' + file)
                    if im is None:
                        count_failed += 1
                        continue
                    count += 1
                    if for_classifier:
                        im = im.astype('float32') / 255
                    if count % 20 == 0:
                        print(str(round(100 * count / (total_paintings - count_failed), 2)) + '%')
                    dataset[artist].append(np.array(im))

        if save_pickle:
            length = len(dataset)
            endpoints = (np.arange(0, NUM_DICTS + 1) * length / NUM_DICTS).astype('uint32')
            # SAVE
            for i in range(NUM_DICTS):
                if for_classifier:
                    dict_dir = DICT_SAVE_DIR + 'full_float_' + str(i) + '.pkl'
                else:
                    dict_dir = DICT_SAVE_DIR + 'full_int_' + str(i) + '.pkl'
                current = dict(list(dataset.items())[endpoints[i]:endpoints[i + 1]])
                with open(dict_dir, 'wb') as f:
                    pickle.dump(current, f)
            if wordy:
                print('Saved!')
    else:
        # LOAD
        dataset = {}
        if for_classifier:
            in_tensors = np.load(DICT_SAVE_DIR + 'in_tensors.npz')['arr_0']
            out_tensors = np.load(DICT_SAVE_DIR + 'out_tensors.npz')['arr_0']
            if wordy:
                print('Loaded!')
            return in_tensors, out_tensors
        else:
            for file in os.listdir(DICT_SAVE_DIR):
                if file.__contains__('full_int'):
                    with open(DICT_SAVE_DIR + file, 'rb') as f:
                        dataset.update(pickle.load(f))
        if wordy:
            print('Loaded!')

    total_paintings = sum(len(dataset[key]) for key in dataset.keys())
    target_height, target_width = get_rescale_dims(dataset, total_paintings,
                                                   rescale_height=rescale_height, rescale_width=rescale_width)

    if for_classifier:
        # Convert to TensorDataset for use with PyTorch classifier by switching (HxWxC) -> (CxHxW)
        images = np.zeros((total_paintings, 3, target_height, target_width), dtype=float)
        labels = []
        count = 0
        for i in range(len(names)):
            for im in dataset[names[i]]:
                images[count, :, :, :] = rescale(im, target_height=target_height, target_width=target_width) \
                    .transpose((2, 0, 1))
                labels.append(i)
                count += 1

        in_tensors = torch.from_numpy(images).view(-1, 3, target_height, target_width)
        if wordy:
            print('Normalizing!')
        for i in range(len(in_tensors)):
            in_tensors[i] = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(in_tensors[i])

        out_tensors = torch.from_numpy(np.asarray(labels)).view(-1)

        return torch.utils.data.TensorDataset(in_tensors, out_tensors)
    else:
        # Convert to dict of tensor images for use with style transfer network by switching (HxWxC) -> (CxHxW)
        for artist in names:
            for i in range(len(dataset[artist])):
                im = rescale(dataset[artist][i],
                             target_height=target_height, target_width=target_width).transpose((2, 0, 1))
                dataset[artist][i] = torch.from_numpy(im).view(3, target_height, target_width)
        return dataset
