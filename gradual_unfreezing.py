import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader, Dataset
import numpy as np
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import time


torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")


def Transforms():
    Train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    Test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return Train_transform, Test_transform


class PetBreedDataset(Dataset):
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
                # labels 1-37 -> 0-36
                label = int(parts[1]) - 1
                img_path = os.path.join(self.images_dir, img)
                self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def getModel():
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    # Freeze all layers to start (same as linear probing)
    for param in model.parameters():
        param.requires_grad = False
    # Replace final layer for 37-class classification
    n_features = model.fc.in_features
    model.fc = nn.Linear(n_features, 37)
    # Only fc is trainable at the start
    for param in model.fc.parameters():
        param.requires_grad = True
    model = model.to(device)
    return model


def unfreeze_layer(model, layer_name, optimizer, lr):
    """
    Unfreeze a named layer and add its parameters to the optimizer
    with the given learning rate.
    """
    layer = getattr(model, layer_name)
    for param in layer.parameters():
        param.requires_grad = True
    # Add new params as a new param group so we can use a different lr if needed
    optimizer.add_param_group({"params": layer.parameters(), "lr": lr})
    print(f"  --> Unfroze {layer_name}")


def evaluation(model, loader):
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
    acc = 100 * corr / total
    return acc


def train(model, train_loader, valid_loader, optimizer, criterion, n_epochs,
          unfreeze_schedule, lr):
    """
    Train with gradual unfreezing.

    unfreeze_schedule: dict mapping epoch number -> layer name to unfreeze
                       e.g. {3: "layer4", 5: "layer3", 7: "layer2", 9: "layer1"}
    Unfreezing happens BEFORE the epoch starts.
    """
    since = time.time()
    best_acc = 0
    best_path = "best_gradual_model.pth"

    for epoch in range(n_epochs):
        # --- Gradual unfreezing: check if a layer should be unfrozen this epoch ---
        if epoch in unfreeze_schedule:
            layer_name = unfreeze_schedule[epoch]
            unfreeze_layer(model, layer_name, optimizer, lr)

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
            torch.save(model.state_dict(), best_path)

        # Show which layers are currently unfrozen
        unfrozen = [n for n, p in model.named_parameters() if p.requires_grad]
        # Summarise to layer-level names for brevity
        unfrozen_layers = sorted({n.split(".")[0] for n in unfrozen})

        print(f"Epoch [{epoch + 1}/{n_epochs}]"
              f"\nUnfrozen layers : {unfrozen_layers}"
              f"\nTrain Loss      = {epoch_loss:.4f}"
              f"\nTrain Accuracy  = {train_acc:.2f}%"
              f"\nValid Accuracy  = {valid_acc:.2f}%\n")

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s  :-)")
    print(f"Best Validation Accuracy = {best_acc:.2f}%")


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
    test_load  = DataLoader(test_set,  batch_size=32, shuffle=False)

    model = getModel()
    criterion = nn.CrossEntropyLoss()
    lr = 0.001
    unfreeze_lr = 0.0001  # for deeper layers

    # Start optimizer with only fc parameters (linear probing phase)
    optimizer = optim.Adam(model.fc.parameters(), lr=lr)

    # --- Unfreeze schedule ---
    # Epoch 0-2  : only fc trained  (linear probing warm-up, 3 epochs)
    # Epoch 3    : unfreeze layer4
    # Epoch 5    : unfreeze layer3
    # Epoch 7    : unfreeze layer2
    # Epoch 9    : unfreeze layer1
    # Total: 12 epochs so every stage gets at least 2 epochs
    # Adjust freely depending on your available compute.
    unfreeze_schedule = {
    3: "layer4",
    6: "layer3",
    9: "layer2",
    12: "layer1",
      }
    n_epochs = 15

    train(model, train_load, valid_load, optimizer, criterion,
          n_epochs, unfreeze_schedule, unfreeze_lr)

    # Load best checkpoint and evaluate on test set
    model.load_state_dict(torch.load("best_gradual_model.pth"))
    test_acc = evaluation(model, test_load)
    print(f"\nTest Accuracy = {test_acc:.2f}%")

if __name__ == "__main__":
    main()