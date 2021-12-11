import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dataset import get_painting_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
# kwargs = {}


def main():
    print('Pre-loading ResNet!')
    net = models.resnet50(pretrained=True, progress=True)
    net.fc = nn.Linear(2048, 50)
    net = net.double().to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    # optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
    optimizer = optim.Adam(net.parameters(), lr=0.01, weight_decay=0.0001)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.3, patience=5)

    NUM_EPOCHS = 200
    BATCH_SIZE = 8
    PRINT_EVERY = 1
    VAL_EVERY = 5
    SAVE_EVERY = 10

    print('Getting dataset!')
    dataset_tensor = get_painting_dataset(for_classifier=True, rescale_height=-1, rescale_width=-1,
                                          use_resized=True, save_pickle=False, load_pickle=True, wordy=True)

    train_size = int(0.75 * len(dataset_tensor))
    val_size = int(0.05 * len(dataset_tensor))
    test_size = len(dataset_tensor) - train_size - val_size
    train, val, test = torch.utils.data.random_split(dataset_tensor, [train_size, val_size, test_size])
    train_loader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(val, batch_size=BATCH_SIZE, shuffle=True, **kwargs)

    train_losses = [[], []]
    print('Training!')
    for epoch in range(NUM_EPOCHS):
        print("========Epoch {}/{}========".format(epoch + 1, NUM_EPOCHS))
        # avg_loss = torch.tensor(0, device=device, dtype=torch.long)
        # running_loss = torch.tensor(0, device=device, dtype=torch.long)
        avg_loss = 0.0
        running_loss = 0.0
        batch_count = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device=device), labels.to(device=device, dtype=torch.long)
            optimizer.zero_grad()
            # print("grabbed batch")

            outputs = net(inputs)
            # print("got outputs")
            loss = criterion(outputs, labels)
            loss_val = loss.cpu().item()
            avg_loss += loss_val
            # print("got loss")

            loss.backward()
            optimizer.step()
            # print("did backprop")

            running_loss += loss_val

            if batch_count % PRINT_EVERY == 0:
                print('\tminibatch: {}\tLoss={}'.format(batch_count, running_loss / PRINT_EVERY))
                # running_loss = torch.tensor(0, device=device)
                running_loss = 0.0
            batch_count += 1

        avg_loss = avg_loss * BATCH_SIZE / train_size
        # scheduler.step(avg_loss)

        if epoch % VAL_EVERY == 0:
            with torch.no_grad():
                num_correct = 0
                for val_inputs, val_labels in val_loader:
                    val_outputs = torch.sigmoid(net(val_inputs.to(device))).cpu()
                    num_correct += torch.sum(torch.argmax(val_outputs, dim=1) == val_labels).item()

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
    main()
