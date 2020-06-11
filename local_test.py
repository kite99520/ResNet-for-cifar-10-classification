import torch.utils.data.dataloader as loader
import torchvision
import torchvision.transforms as f_trs
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
import torch

# from modelarts.session import Session


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Block, self).__init__()
        self.sets = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.sets(x) + self.shortcut(x))


class Net(nn.Module):
    def __init__(self, block, num_block, num_classes=10):
        super(Net, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def main():

    trans2 = f_trs.Compose([f_trs.ToTensor(), f_trs.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    path_2 = r"D:\personal_project\datas"
    test_set = torchvision.datasets.CIFAR10(root=path_2, train=False, download=False,
                                            transform=trans2)
    test_loader = loader.DataLoader(test_set, batch_size=100, shuffle=True, num_workers=3)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    """process 2 network"""
    net = Net(Block, [2, 2, 2, 2])
    net.load_state_dict(torch.load(r"D:\personal_project\params.pkl", map_location="cpu"))


    """process 4 test"""
    corr = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            # images = images.to(device)
            # labels = labels.to(device)
            outputs = net(images)
            a, predict = torch.max(outputs.data, 1)
            total += labels.size(0)
            corr += (predict == labels).sum().item()
    print("Accuracy of the network on the 10000 test images: %d %%" % (100 * corr / total))

    class_corr = torch.zeros(10).tolist()
    class_total = torch.zeros(10).tolist()
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = net(images)
            a, predict = torch.max(outputs.data, 1)
            for i in range(5):
                class_total[labels[i]] += 1
                class_corr[labels[i]] += (predict[i] == labels[i])
    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_corr[i] / class_total[i]))

if __name__ == '__main__':
    main()