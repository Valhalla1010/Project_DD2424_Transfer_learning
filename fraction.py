import torch
import torch.nn as nn 
import torch.optim as optim 
from torch.utils.data import random_split, DataLoader, Subset
from torchvision.models import resnet18, ResNet18_Weights
from collections import defaultdict
import random
import time
from linearprobing import PetBreedDataset, Transforms, evaluation

torch.manual_seed(42)
random.seed(42)
device = torch.device("cpu")

# Finetuning w/ 100%, 10%, 1% of the data

# random sampling w/ at least 1 sample kept per class
def stratifiedSubset(dataset, fraction):
    by_label = defaultdict(list)
    for idx, (_, label) in enumerate(dataset.samples):
        by_label[label].append(idx)
    selected = []
    for idxs in by_label.values():
        k = max(1, int(len(idxs) * fraction))
        selected.extend(random.sample(idxs, k))
    return selected


def getModel():
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer4.parameters():
        param.requires_grad = True
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
    if best_acc == 0:
        torch.save(model.state_dict(), save_path)
    total_time = time.time() - since
    print(f"Training complete in {total_time // 60:.0f}m {total_time % 60:.0f}s  :-) ")
    print(f"Best Validation Accuracy= {best_acc:.2f}%")


def main():
    root = "Datasets"
    train_transform, test_transform = Transforms()

    full_dataset = PetBreedDataset(root, "trainval.txt", train_transform)
    test_set = PetBreedDataset(root, "test.txt", test_transform)
    test_load = DataLoader(test_set, batch_size=32, shuffle=False)

    fractions = [1.0, 0.1, 0.01]

    for frac in fractions:
        print(f"\nTraining with {int(frac * 100)}% of data")
        idxs = stratifiedSubset(full_dataset, frac)
        subset = Subset(full_dataset, idxs)

        train_size = int(0.8 * len(subset))
        valid_size = len(subset) - train_size
        train_data, valid_data = random_split(subset, [train_size, valid_size])

        train_load = DataLoader(train_data, batch_size=32, shuffle=True)
        valid_load = DataLoader(valid_data, batch_size=32, shuffle=False)

        model = getModel()
        criterion = nn.CrossEntropyLoss()
        lr = 0.0001
        n_epochs = 15
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.Adam(trainable, lr)
        save_path = f"best_frac{int(frac * 100)}.pth" # pytorch saved

        train(model, train_load, valid_load, optimizer, criterion, n_epochs, save_path)
        model.load_state_dict(torch.load(save_path))
        test_acc = evaluation(model, test_load)
        print(f"Test Accuracy = {test_acc:.2f}%")

if __name__ == "__main__":
    main()