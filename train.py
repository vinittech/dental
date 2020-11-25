import os
import argparse
import time
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms

from model import VGG as CNN

parser = argparse.ArgumentParser(description='Image Classification Task')
parser.add_argument('--batchSize', type=int, default=64, help='Training batch size')
parser.add_argument('--totalEpochs', type=int, default=25, help='Number of epochs to train for')
parser.add_argument('--snapshots', type=int, default=10, help='Snapshot Frequency of the checkpoint')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning Rate')
parser.add_argument('--threads', type=int, default=2, help='Number of threads for data loader to use')
parser.add_argument('--rootDir', type=str, default='Dataset/', help='Directory path to root of the training data')
parser.add_argument('--saveCheckpoint', type=str, default='checkpoint/', help='Directory to store checkpoint')

opt = parser.parse_args()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

TRANSFORM_IMG = transforms.Compose([
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),
        transforms.ToTensor()
    ])
dataset = torchvision.datasets.ImageFolder(root=opt.rootDir, transform=TRANSFORM_IMG)
train_set, validation_set = torch.utils.data.random_split(dataset, [int(len(dataset)*0.8), len(dataset) - int(len(dataset)*0.8)])
classes = dataset.classes

train_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize,shuffle=False)
validation_loader = DataLoader(dataset=validation_set, num_workers=opt.threads, batch_size=opt.batchSize,shuffle=False)

train_accuracy = []
validation_accuracy = []

model = CNN(num_classes=len(dataset.classes)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr,weight_decay=0.0001)

def train(epoch):
    epoch_loss = 0
    train_acc = 0

    model.train()
    for iteration, (image,label) in enumerate(train_loader):
        images = image.to(device)
        labels = label.to(device)

        t0 = time.time()
        prediction = model(images)
        loss = criterion(prediction, labels)
        t1 = time.time()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        _, prediction = torch.max(prediction.data, 1)
        train_acc += torch.sum(prediction.data == labels.data).cpu().numpy()

        print("===> Epoch[{}]({}/{}): Loss: {:.4f} || Timer: {:.4f} sec.".format(epoch, iteration,
                                                                                 len(train_loader), epoch_loss,
                                                                                 (t1 - t0)))
    epoch_acc = (train_acc/len(train_set))*100


    train_accuracy.append(epoch_acc)
    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(train_set)))
    print("===> Epoch {} Complete: Accuracy: {:.4f}".format(epoch, epoch_acc))


def validation(epoch):
    with torch.no_grad():
        validation_acc = 0
        for iteration, (image, label) in enumerate(validation_loader):
            images = image.to(device)
            labels = label.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            validation_acc += (predicted == labels).sum().item()
        
        epoch_acc = (validation_acc/len(validation_set))*100
        validation_accuracy.append(epoch_acc)
        print('Accuracy of the model on the {} test images: {} %'.format(len(validation_set), epoch_acc))


def checkpoint(epoch):
    if epoch % opt.snapshots == 0:
        model_out_path = opt.saveCheckpoint + "epoch_{}.pth".format(epoch)
        torch.save(model.state_dict(), model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

for epoch in range(opt.totalEpochs):
    train(epoch)
    validation(epoch)
    checkpoint(epoch)

plt.plot(range(opt.totalEpochs),train_accuracy)
plt.xlabel('Epochs')
plt.ylabel('Train Accuracy')
plt.savefig('train_acc.png')
plt.clf()

plt.plot(range(opt.totalEpochs),validation_accuracy)
plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy')
plt.savefig('val_acc.png')





