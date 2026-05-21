import os
import torch
import torch.nn as nn 
import torch.optim as optim 
from torch.utils.data import random_split, DataLoader
from torchvision.models import resnet18, ResNet18_Weights
import time
from linearprobing import PetBreedDataset, Transforms, evaluation

torch.manual_seed(42)
device = torch.device("cpu")

backbone_layers = ["layer1", "layer2", "layer3", "layer4"]

def getModel(l):
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False # freeze

    # unfreeze last l layers
    for name in backbone_layers[-l:]:
        for param in getattr(model, name).parameters():
            param.requires_grad = True

    # replace final layer
    n_features = model.fc.in_features
    model.fc = nn.Linear(n_features, 37)
    model = model.to(device)
    
    return model


def train(model, train_loader, valid_loader, optimizer, criterion, n_epochs, save_path):
    since = time.time()
    best_acc = 0
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
        valid_acc = evaluation(model, valid_loader)
        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save(model.state_dict(), save_path)
        print(f"Epoch [{epoch + 1}/{n_epochs}]"
              f"\nTrain Loss = {epoch_loss:.4f}"
              f"\nTrain Accuracy = {train_acc:.2f}%"
              f"\nValid Accuracy = {valid_acc:.2f}%")
    total_time = time.time() - since
    print(f"Training complete in {total_time // 60:.0f}m {total_time % 60:.0f}s  :-) ")
    print(f"Best Validation Accuracy= {best_acc:.2f}%")


def main():
    root = "Datasets"
    train_transform, test_transform = Transforms()

    dataset = PetBreedDataset(root, "trainval.txt", train_transform)
    test_set = PetBreedDataset(root, "test.txt", test_transform)

    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size
    train_data, valid_data = random_split(dataset, [train_size, valid_size])

    train_load = DataLoader(train_data, batch_size=32, shuffle=True)
    valid_load = DataLoader(valid_data, batch_size=32, shuffle=False)
    test_load = DataLoader(test_set, batch_size=32, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    lr = 0.0001
    n_epochs = 15

    # fine-tune last l layers for l = 1, 2, 3, 4
    for l in range(1, 5):
        print(f"\nl = {l}, unfreezing: {backbone_layers[-l:]} + fc")
        model = getModel(l)
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.Adam(trainable, lr)
        save_path = f"best_finetune_l{l}.pth"
        train(model, train_load, valid_load, optimizer, criterion, n_epochs, save_path)
        model.load_state_dict(torch.load(save_path))
        test_acc = evaluation(model, test_load)
        print(f"Test Accuracy (l={l}) = {test_acc:.2f}%")

if __name__ == "__main__":
    main()