# David Marzban
# DD2424
# Deep Learning in Data Science
# 2026-05-19
""" 
    Fine-tuning with limited data
"""
import matplotlib.pyplot as plt
import os
import time
import torch
import torch.nn as nn 
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.model_selection import train_test_split
from PIL import Image
from linearprobing import PetBreedDataset, getModel, evaluation, Transforms

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Add augmentation
def Transform_limited():
    Train_tranforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    Test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return Train_tranforms, Test_transform


def train_limited(model, train_loader, valid_loader ,optimizer, criterion, n_epochs):
    since = time.time()
    best_acc = 0
    best_path = "best_limited.pth"
    train_history = []
    valid_history = []
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
        train_history.append(train_acc)
        valid_history.append(valid_acc)
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

    return train_history, valid_history

def plot_learning_curves(train_acc,valid_acc):
    epochs=range(1,len(train_acc)+1)
    plt.figure()
    plt.plot(epochs,train_acc,label="Train Accuracy")
    plt.plot(epochs,valid_acc,label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.show()

def plot_fraction_results(fractions, accuracies):
    plt.figure()
    plt.plot(fractions,accuracies,marker="o")
    plt.xlabel("Training Data Fraction")
    plt.ylabel("Test Accuracy (%)")
    plt.xticks(fractions)
    plt.show()


def main():
    root = "Datasets"
    train_tranform, test_tranform = Transform_limited()
    #train_tranform, test_tranform = Transforms()

    dataset = PetBreedDataset(root, "trainval.txt", train_tranform)
    test_set = PetBreedDataset(root, "test.txt", test_tranform)

    # split train and validation
    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size
    train_data, valid_data = random_split(dataset, [train_size, valid_size])
    # 100%, 10%, 1%
    fraction = 1.0
    indices = list(range(len(train_data)))
    labels = [train_data.dataset.samples[i][1] for i in train_data.indices]
    # for 100% and 10% 
    if fraction < 1.0:
        selected, _ = train_test_split(
            indices,
            train_size=fraction,
            stratify=labels,
            random_state=42
        )
        train_data = Subset(train_data, selected)
    else:
        train_data = train_data
    
    # for 1%
    #n_samples = max(int(fraction * len(indices)), len(set(labels)))
    #selected, _ = train_test_split(indices, train_size=n_samples, stratify=labels, random_state=42)
    #train_data = Subset(train_data, selected)

    train_load = DataLoader(train_data, batch_size=32, shuffle=True)
    valid_load = DataLoader(valid_data, batch_size=32, shuffle=False)
    test_load  = DataLoader(test_set, batch_size=32, shuffle=False)

    model = getModel()
    criterion = nn.CrossEntropyLoss()
    n_epochs = 15
    # L2 regularization
    optimizer=optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, weight_decay=0.0001)

    # without L2
    #optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

    train_acc, valid_acc = train_limited(model, train_load, valid_load, optimizer, criterion ,n_epochs)
    model.load_state_dict(torch.load("best_limited.pth"))

    test_acc = evaluation(model, test_load)
    print(f"\nFraction = {fraction}")
    print(f"Test Accuracy = {test_acc:.2f}%")
    plot_learning_curves(train_acc, valid_acc)


    """fractions=[1, 0.1, 0.01]
    accuracies=[ # change accuray 
        0.873,
        0.78,
        0.3123
    ]
    plot_fraction_results(
        fractions,
        accuracies
    )"""

if __name__ == "__main__":
    main()
