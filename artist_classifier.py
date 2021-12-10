import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dataset import get_dataset
from resnet_pytorch import ResNet
# from ResNet import Bottleneck, ResNet, ResNet50

dataset_tensor, classes = get_dataset(rescale_height=-1, rescale_width=-1, use_resized=True, save_pickle=False, load_pickle=True,
                wordy=True)

print('x')
train_size = int(0.7 * len(dataset_tensor))
val_size = int(0.1 * len(dataset_tensor))
test_size = len(dataset_tensor) - train_size - val_size

train, val, test = torch.utils.data.random_split(dataset_tensor, [train_size, val_size, test_size])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
kwargs = {'num_workers': 2, 'pin_memory': True} if torch.cuda.is_available() else {}

trainloader = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True, **kwargs)

net = ResNet.from_pretrained('resnet50', 50)
print('res')
# net = ResNet50(50).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.1, patience=5)

EPOCHS = 100
for epoch in range(EPOCHS):
    losses = []
    running_loss = 0
    for i, inp in enumerate(trainloader):
        inputs, labels = inp
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % 1 == 0 and i > 0:
            print(f'Loss [{epoch + 1}, {i}](epoch, minibatch): ', running_loss / 1)
            running_loss = 0.0

    avg_loss = sum(losses) / len(losses)
    scheduler.step(avg_loss)

print('Training Done')