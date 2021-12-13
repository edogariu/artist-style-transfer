import random

import fastai.vision.learner
import matplotlib.pyplot as plt
import numpy as np
import pandas
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

from fastai.vision import *
import fastai
# from fastai.callbacks import *
# from fastai.callbacks.hooks import *
from PIL import *

from dataset import get_painting_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
kwargs = {'num_workers': 4, 'pin_memory': False} if torch.cuda.is_available() else {}


class CustomDataLoader:
    def __init__(self, data_x, data_y, batch_size, shuffle):
        self.data_x = data_x
        self.data_y = data_y
        assert len(data_x) == len(data_y), "Inputs and outputs should have same number of datapoints! :)"
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.shape = data_x.shape
        self.indices = np.arange(0, self.shape[0])
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.count = 0

    def __iter__(self):
        self.indices = np.arange(0, self.shape[0])
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.count = 0
        return self

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, item):
        return torch.from_numpy(self.data_x[self.indices[item]]).view(self.shape[1], self.shape[2], self.shape[3]), \
               torch.from_numpy(self.data_y[self.indices[item]])

    def __next__(self):
        if self.count < self.shape[0] // self.batch_size:
            self.count += 1
            # return self.dataset[self.indices[self.batch_size * (self.count - 1): self.batch_size * self.count]]
            # in_tensors = torch.gather(self.dataset[:][0], 0,
            #                           self.indices[self.batch_size * (self.count - 1): self.batch_size * self.count])
            # out_tensors = torch.gather(self.dataset[:][1], 0,
            #                            self.indices[self.batch_size * (self.count - 1): self.batch_size * self.count])
            batch_indices = self.indices[self.batch_size * (self.count - 1): self.batch_size * self.count]
            return torch.from_numpy(self.data_x[batch_indices]). \
                       view(-1, self.shape[1], self.shape[2], self.shape[3]).to(device), \
                   torch.from_numpy(self.data_y[batch_indices]).view(-1).to(device, dtype=torch.long)
        else:
            raise StopIteration


def train_classifier(resnet_size=34):
    print('Pre-loading ResNet!')
    if resnet_size == 18:
        net = models.resnet18(pretrained=True, progress=True)
    elif resnet_size == 34:
        net = models.resnet34(pretrained=True, progress=True)
    elif resnet_size == 50:
        net = models.resnet50(pretrained=True, progress=True)
        net.fc = nn.Sequential(nn.BatchNorm1d(2048), nn.Linear(2048, 512), nn.BatchNorm1d(512), nn.Linear(512, 19))

    else:
        print('Please enter valid resnet size :)')
        return -1
    if resnet_size == 50:
        print('gotta finish this')
        # learn = cnn_learner(None, models.resnet50, metrics=error_rate)
        # learn.load('models/best-2.pth')
        # sd = learn.model.state_dict()
        # net.load_state_dict(sd, strict=False)
    else:
        net.fc = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 50))
        for param in net.parameters():
            param.requires_grad = False
        for name, param in net.named_parameters():
            if name == 'fc.0.weight' or name == 'fc.0.bias' or name == 'fc.2.weight' or name == 'fc.2.bias':
                param.requires_grad = True
            else:
                param.requires_grad = False
    net = net.double().to(device)

    print('Loading Dataset!')
    data_x, data_y = get_painting_dataset(for_classifier=True, rescale_height=-1, rescale_width=-1, use_resized=True,
                                          save_pickle=False, load_pickle=True, wordy=True)
    # data_x, data_y = np.ones((8000, 3, 256, 224)), np.ones(8000)
    artists = ['Vincent_van_Gogh', 'Edgar_Degas', 'Pablo_Picasso', 'Pierre-Auguste_Renoir', 'Paul_Gauguin', 'Francisco_Goya',
       'Rembrandt', 'Alfred_Sisley', 'Titian', 'Marc_Chagall', 'Rene_Magritte', 'Amedeo_Modigliani', 'Paul_Klee',
       'Henri_Matisse', 'Andy_Warhol', 'Mikhail_Vrubel', 'Sandro_Botticelli', 'Leonardo_da_Vinci', 'Peter_Paul_Rubens']

    # Replace spaces with _
    df = pandas.read_csv('images/archive/artists.csv')
    names = list(df['name'])
    for i in range(len(names)):
        names[i] = names[i].replace(' ', '_')

    print('Creating tensors!')
    in_tensors = []
    out_tensors = []
    for i in range(len(data_x)):
        name = names[int(data_y[i])]
        try:
            index = artists.index(name)
        except ValueError as e:
            continue
        in_tensors.append(torch.from_numpy(data_x[i]).unsqueeze(0))
        out_tensors.append(index)

    print(in_tensors[0].size(), out_tensors[0])
    num_correct = 0
    net.eval()
    with torch.no_grad():
        for i in range(len(in_tensors)):
            tensor = in_tensors[i].to(device=device, dtype=torch.double)
            output = torch.softmax(net(tensor).squeeze(), dim=0)
            pred = torch.argmax(output).cpu().item()
            print(pred, out_tensors[i], output)
            if out_tensors[i] == pred:
                num_correct += 1
    print('Acc = {}', round(100 * num_correct / len(out_tensors), 2))

    return 0






    criterion = nn.CrossEntropyLoss().to(device)
    # optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
    optimizer = optim.Adam(net.parameters(), lr=0.05, weight_decay=0.0001)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.3, patience=5)

    NUM_EPOCHS = 200
    BATCH_SIZE = 8
    PRINT_EVERY = 16
    VAL_EVERY = 2
    SAVE_EVERY = 5

    print('Loading dataset!')
    data_x, data_y = get_painting_dataset(for_classifier=True, rescale_height=-1, rescale_width=-1,
                                          use_resized=True, save_pickle=False, load_pickle=True, wordy=True)
    # data_x, data_y = np.ones((8000, 3, 256, 224)), np.ones(8000)

    length = len(data_x)
    train_size = int(0.75 * length)
    val_size = int(0.05 * length)
    random_indices = np.arange(0, length)
    np.random.shuffle(random_indices)
    print('Splitting dataset: we have {} training datapoints!'.format(train_size))
    train_x, train_y = data_x[random_indices[0:train_size]], \
                       data_y[random_indices[0:train_size]]
    val_x, val_y = data_x[random_indices[train_size:val_size + train_size]], \
                   data_y[random_indices[train_size:val_size + train_size]]
    test_x, test_y = data_x[random_indices[val_size + train_size:length]], \
                     data_y[random_indices[val_size + train_size:length]]

    # train, val, test = torch.utils.data.random_split(dataset_tensor, [train_size, val_size, test_size])
    #
    # train_loader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
    # val_loader = torch.utils.data.DataLoader(val, batch_size=BATCH_SIZE, shuffle=True, **kwargs)

    print('Getting DataLoaders!')
    train_loader = CustomDataLoader(data_x=train_x, data_y=train_y, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = CustomDataLoader(data_x=val_x, data_y=val_y, batch_size=BATCH_SIZE, shuffle=True)

    train_losses = [[], []]
    print('Training!')
    for epoch in range(NUM_EPOCHS):
        print("========Epoch {}/{}========".format(epoch + 1, NUM_EPOCHS))
        avg_loss = torch.zeros(1).double().to(device)
        batch_count = 0
        batch_start = time.thread_time_ns()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            # torch.cuda.synchronize()
            # print("grabbed batch\t", time.thread_time_ns() - batch_start)

            outputs = net(inputs)
            # torch.cuda.synchronize()
            # print("got outputs\t", time.thread_time_ns() - batch_start)
            loss = criterion(outputs, labels)
            avg_loss += loss
            # torch.cuda.synchronize()
            # print("got loss\t", time.thread_time_ns() - batch_start)

            loss.backward()
            optimizer.step()
            # torch.cuda.synchronize()
            # print("did backprop\t", time.thread_time_ns() - batch_start)

            if batch_count % PRINT_EVERY == 0:
                print('\tminibatch: {}\tLoss={}\ttime={}'.format(batch_count, loss.cpu().item(),
                                                                 time.thread_time_ns() - batch_start))
            batch_count += 1

            del inputs
            del labels
            del outputs
            del loss

            batch_start = time.thread_time_ns()

        avg_loss = avg_loss.cpu().item() * BATCH_SIZE / train_size
        # scheduler.step(avg_loss)

        if epoch % VAL_EVERY == 0:
            with torch.no_grad():
                num_correct = 0
                for val_inputs, val_labels in val_loader:
                    val_outputs = torch.sigmoid(net(val_inputs)).cpu()
                    num_correct += torch.sum(torch.argmax(val_outputs, dim=1) == val_labels.cpu()).item()

            print('Training Loss={}\tVal Acc={}\n'.format(avg_loss, round(100 * num_correct / val_size, 2)))
            train_losses[0].append(avg_loss)
            train_losses[1].append(100 * num_correct / val_size)
        else:
            train_losses[0].append(avg_loss)
            train_losses[1].append(-1.0)

        if epoch % SAVE_EVERY == 0:
            torch.save(net.state_dict(), 'models/classifier_{}.pth'.format(epoch))
            np.save('models/classifier_losses_{}.npy'.format(epoch), np.array(train_losses))

    print('Training Done')


if __name__ == '__main__':
    train_classifier(resnet_size=50)
