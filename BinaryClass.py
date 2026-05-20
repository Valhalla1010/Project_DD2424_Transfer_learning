# David Marzban
# DD2424
# Deep Learning in Data Science
# 2026-05-16
""" check replace the final layer of 
    the pre-trained ConvNet to solve the
    binary classification problem 
    of recognising pictures of Dog Vs Cat
"""


import os
import torch
import torch.nn as nn 
import torch.optim as optim 
from torch.utils.data import random_split, DataLoader, Dataset
import numpy as py
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import time



torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")


def Transforms():
    # Transform
    Train_transform = transforms.Compose([
        #resizing to 224 by 224
        transforms.Resize((224, 224)),
        #Horizontally flip the given image randomly
        transforms.RandomHorizontalFlip(),
        #Convert a PIL Image or ndarray to tensor 
        transforms.ToTensor(),
        #Normalize a tensor image with mean and standard deviation.
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
   
    Test_transform = transforms.Compose([
        # resize echa input image
        transforms.Resize(256),
        #Crops the given image at the center.
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return Train_transform, Test_transform

# load Data 
class BinaryPetDataset(Dataset):
    def __init__(self, root, split, transform=None):
        self.root = root
        self.transform = transform
        self.images_dir = os.path.join(root, "images")
        self.split = os.path.join(root, "annotations", split)
        self.samples = []
        with open(self.split, "r") as f:
            for line in f:
                parts = line.strip().split()
                img = parts[0] + ".jpg"
                # Binary classification
                species  = int(parts[2]) # parts[2] => Binary cat/dog 
                # species cat = 1 & dog = 2, convert to binary (cat=0, dog=1)
                label = 0 if species == 1 else 1
                img_path = os.path.join(self.images_dir, img)
                self.samples.append((img_path, label))

    def __len__(self):
        result = len(self.samples)
        return result

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# get model and Set up Pytorch with ResNet-18
def getModel():
    model = resnet18(weights = ResNet18_Weights.DEFAULT)
    # freeze all pre-trained 
    for param in model.parameters():
        param.requires_grad = False

    # unfreeze
    #for param in model.layer4.parameters():
        #param.requires_grad = True
    # replace final classifier
    n_features = model.fc.in_features
    model.fc = nn.Linear(n_features, 2)
    
    model = model.to(device)
    return model

# evaluation
def evaluation(model, loader):
    # Set the module in evaluation mode.
    model.eval()
    corr = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            corr += (preds == labels).sum().item()
    acc = 100 * corr/total
    return acc


def train(model, train_loader, valid_loader ,optimizer, criterion, n_epochs):
    since = time.time()
    best_acc = 0
    best_path = "best_model.pth"
    for epoch in range(n_epochs):
        model.train()
        corr = 0
        total = 0
        run_loss = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            corr += (preds == labels).sum().item()
        epoch_loss = run_loss / total
        train_acc = 100 * corr / total
        # validation
        valid_acc = evaluation(model, valid_loader)
        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save(model.state_dict(), best_path)
        print(f"Epoch [{epoch + 1}/{n_epochs}]"
              f"\nTrain Loss = {epoch_loss:.4f}"
              f"\nTrain Accuracy = {train_acc:.2f}%"
              f"\nValid Accuracy = {valid_acc:.2f}%")
    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s  :-) ")
    print(f"Best Validation Accuracy= {best_acc:.2f}%")


def main():
    root = "Datasets"
    train_trasform, test_transform = Transforms()
    # load data
    dataset = BinaryPetDataset(root,  "trainval.txt", train_trasform)
    test_set = BinaryPetDataset(root, "test.txt", test_transform)
    # split train & validation
    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size

    train_data, valid_data = random_split(dataset, [train_size, valid_size])

    train_load = DataLoader(train_data, batch_size=32, shuffle=True)
    valid_load = DataLoader(valid_data, batch_size=32, shuffle=False)
    test_load = DataLoader(test_set, batch_size=32, shuffle=False)

    model = getModel()
    criterion = nn.CrossEntropyLoss()
    lr = 0.001
    optimizer = optim.Adam(model.fc.parameters(), lr)
    
    n_epochs = 15

    train(model, train_load, valid_load, optimizer, criterion, n_epochs)
    model.load_state_dict(torch.load("best_model.pth"))

    test_acc = evaluation(model, test_load)
    print(f"Test Accuracy = {test_acc:.2f}%")

if __name__ == "__main__":
    main()
