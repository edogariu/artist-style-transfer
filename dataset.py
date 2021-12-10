import numpy as np
import cv2
import pandas
import pickle
import torch
import os
import matplotlib.pyplot as plt
import torchvision.transforms as transforms


# Script to load TensorDataset dataset of paintings with artist labels (from 0 to 49)
def get_dataset(ret_avg_dataset=False, rescale_height=-1, rescale_width=-1, use_resized=True,
                save_pickle=False, load_pickle=True, wordy=False):
    df = pandas.read_csv('archive/artists.csv')

    # Replace spaces with _
    names = list(df['name'])
    for i in range(len(names)):
        names[i] = names[i].replace(' ', '_')

    # Get images for each artist
    dataset = {artist: [] for artist in names}
    if ret_avg_dataset:
        average_image = {artist: [] for artist in names}

    count = 0
    total_paintings = np.sum(df['paintings'])
    if not load_pickle:
        if use_resized:
            for i in range(len(dataset.keys())):
                num_paintings = df['paintings'][i]  # Number of paintings from each artist
                for num in range(1, num_paintings + 1):
                    im = cv2.imread('archive/resized/resized/' + names[i] + '_' + str(num) + '.jpg')
                    if im is None:
                        continue
                    im = im.astype('float32')
                    count += 1
                    if count % 20 == 0 and wordy:
                        print(str(round(100 * count / total_paintings, 2)) + '%')
                    dataset[names[i]].append(np.array(im))
        else:
            for artist in names:
                dir = 'archive/images/images/' + artist
                for file in os.listdir(dir):
                    im = cv2.imread(dir + '/' + file)
                    count += 1
                    if count % 20 == 0:
                        print(str(round(100 * count / total_paintings, 2)) + '%')
                    dataset[artist].append(np.array(im))

        if save_pickle:
            # SAVE
            with open('saved_dictionary.pkl', 'wb') as f:
                pickle.dump(dataset, f)
            if wordy:
                print('Saved!')
    else:
        # LOAD
        with open('saved_dictionary.pkl', 'rb') as f:
            dataset = pickle.load(f)
        if wordy:
            print('Loaded!')

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
        target_height = avg_height
        target_width = avg_width
    else:
        target_height = rescale_height
        target_width = rescale_width

    # Rescale each image to target dimensions
    for artist in names:
        total_paintings_artist = len(dataset[artist])
        # average_image[artist] = np.zeros((target_height, target_width, 3))
        for i in range(total_paintings_artist):
            image = dataset[artist][i]

            # Source width and height in pixels
            src_width = image.shape[1]
            src_height = image.shape[0]

            # Scaling parameters
            w_s = target_width / src_width
            h_s = target_height / src_height

            # Affine matrices for width and height
            Affine_Mat_w = [w_s, 0, target_width / 2 - w_s * src_width / 2]
            Affine_Mat_h = [0, h_s, target_height / 2 - h_s * src_height / 2]
            M = np.c_[Affine_Mat_w, Affine_Mat_h].T

            dataset[artist][i] = cv2.warpAffine(image, M, (target_width, target_height))
            if ret_avg_dataset:
                average_image[artist] += dataset[artist][i] / total_paintings_artist

    if wordy:
        print('Rescaled!')

    # Convert to TensorDataset for use with PyTorch
    images = []
    labels = []
    for i in range(len(names)):
        for im in dataset[names[i]]:
            images.append(np.asarray(im).transpose((1, 2, 0)))
            labels.append(i)

    # Do same thing with average images
    if ret_avg_dataset:
        avg_img = []
        label_avg_img = []
        for i in range(len(names)):
            avg_img.append(np.asarray(average_image[names[i]]).transpose((1, 2, 0)))
            label_avg_img.append(i)

    in_tensors = torch.from_numpy(np.array(images)).view(-1, 3, target_height, target_width)
    out_tensors = torch.from_numpy(np.array(labels)).view(-1, 1)

    return torch.utils.data.TensorDataset(in_tensors, out_tensors), dataset.keys()


if __name__ == '__main__':
    get_dataset(ret_avg_dataset=False, rescale_height=-1, rescale_width=-1, use_resized=True,
                save_pickle=True, load_pickle=False,
                wordy=True)
