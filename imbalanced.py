import os
import torch
import torch.nn as nn 
import torch.optim as optim 
from torch.utils.data import random_split, DataLoader, WeightedRandomSampler, Dataset
import numpy as np
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
from sklearn.metrics import f1_score, classification_report
import random
import time

torch.manual_seed(42)
random.seed(42)
device = torch.device("cpu")


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


class PetDataset(Dataset):
    def __init__(self, root, split, transform=None):
        self.transform = transform
        self.images_dir = os.path.join(root, "images")
        self.samples = []
        with open(os.path.join(root, "annotations", split), "r") as f:
            for line in f:
                parts = line.strip().split()
                img = parts[0] + ".jpg"
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


class ImbalancedPetDataset(Dataset):
    def __init__(self, root, split, transform=None, cat_frac=0.2):
        self.transform = transform
        self.images_dir = os.path.join(root, "images")
        self.samples = []

        dog_samples = []
        cat_by_class = {}
        with open(os.path.join(root, "annotations", split), "r") as f:
            for line in f:
                parts = line.strip().split()
                img = parts[0] + ".jpg"
                label = int(parts[1]) - 1
                species = int(parts[2])  # 1 = cat, 2 = dog
                img_path = os.path.join(self.images_dir, img)
                if species == 2:
                    dog_samples.append((img_path, label))
                else:
                    cat_by_class.setdefault(label, []).append((img_path, label))

        cat_samples = []
        for cls_samples in cat_by_class.values():
            k = max(1, int(len(cls_samples) * cat_frac))
            cat_samples.extend(random.sample(cls_samples, k))

        self.samples = dog_samples + cat_samples

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
    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer4.parameters():
        param.requires_grad = True
    n_features = model.fc.in_features
    model.fc = nn.Linear(n_features, 37)
    model = model.to(device)
    return model


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
    return 100 * corr / total


def evaluationDetailed(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    per_class_acc = []
    for c in range(37):
        mask = all_labels == c
        if mask.sum() == 0:
            per_class_acc.append(float('nan'))
        else:
            per_class_acc.append(100 * (all_preds[mask] == all_labels[mask]).mean())

    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    return per_class_acc, macro_f1, weighted_f1


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
    print(f"Training Completed in {total_time // 60:.0f}m {total_time % 60:.0f}s  :-) ")
    print(f"Best Validation Accuracy= {best_acc:.2f}%")


def getClassWeights(dataset):
    counts = np.zeros(37)
    for _, label in dataset.samples:
        counts[label] += 1
    w = 1.0 / np.where(counts == 0, 1, counts)
    w = w / w.sum() * 37
    return torch.FloatTensor(w).to(device)


def getSampler(dataset, indices):
    counts = np.zeros(37)
    for _, label in dataset.samples:
        counts[label] += 1
    sample_weights = np.array([1.0 / counts[label] for _, label in dataset.samples])
    sample_weights = sample_weights[indices]
    return WeightedRandomSampler(torch.DoubleTensor(sample_weights), len(sample_weights), replacement=True)


def main():
    root = "Datasets"
    train_transform, test_transform = Transforms()
    n_epochs = 15

    test_set = PetDataset(root, "test.txt", test_transform)
    test_load = DataLoader(test_set, batch_size=32, shuffle=False)

    # corresponding breeds to class IDs
    cat_classes = {0, 5, 6, 7, 9, 11, 17, 20, 23, 26, 27, 32, 33}

    # experiment 1: imbalanced, no fix
    print("\nExp 1: Imbalanced (no fix)")
    dataset1 = ImbalancedPetDataset(root, "trainval.txt", train_transform, cat_frac=0.2)
    train_size = int(0.8 * len(dataset1))
    valid_size = len(dataset1) - train_size
    train_data, valid_data = random_split(dataset1, [train_size, valid_size])
    train_load = DataLoader(train_data, batch_size=32, shuffle=True)
    valid_load = DataLoader(valid_data, batch_size=32, shuffle=False)
    model1 = getModel()
    criterion1 = nn.CrossEntropyLoss()
    lr = 0.0001
    optimizer1 = optim.Adam([p for p in model1.parameters() if p.requires_grad], lr)
    train(model1, train_load, valid_load, optimizer1, criterion1, n_epochs, "best_imbal.pth")
    model1.load_state_dict(torch.load("best_imbal.pth"))
    test_acc1 = evaluation(model1, test_load)
    per_cls1, macro_f1_1, weighted_f1_1 = evaluationDetailed(model1, test_load)
    print(f"Test Accuracy = {test_acc1:.2f}%")
    print(f"Macro F1 = {macro_f1_1:.4f}  Weighted F1 = {weighted_f1_1:.4f}")
    print("Per-class accuracy:")
    for i, acc in enumerate(per_cls1):
        tag = " <- cat" if i in cat_classes else ""
        print(f"  Class {i+1}: {acc:.1f}%{tag}")

    # experiment 2: weighted cross entropy
    print("\nExp 2: Weighted CrossEntropy")
    dataset2 = ImbalancedPetDataset(root, "trainval.txt", train_transform, cat_frac=0.2)
    class_weights = getClassWeights(dataset2)
    train_size = int(0.8 * len(dataset2))
    valid_size = len(dataset2) - train_size
    train_data, valid_data = random_split(dataset2, [train_size, valid_size])
    train_load = DataLoader(train_data, batch_size=32, shuffle=True)
    valid_load = DataLoader(valid_data, batch_size=32, shuffle=False)
    model2 = getModel()
    criterion2 = nn.CrossEntropyLoss(weight=class_weights)
    optimizer2 = optim.Adam([p for p in model2.parameters() if p.requires_grad], lr)
    train(model2, train_load, valid_load, optimizer2, criterion2, n_epochs, "best_weighted.pth")
    model2.load_state_dict(torch.load("best_weighted.pth"))
    test_acc2 = evaluation(model2, test_load)
    per_cls2, macro_f1_2, weighted_f1_2 = evaluationDetailed(model2, test_load)
    print(f"Test Accuracy = {test_acc2:.2f}%")
    print(f"Macro F1 = {macro_f1_2:.4f}  Weighted F1 = {weighted_f1_2:.4f}")

    # experiment 3: oversampling
    print("\nExp 3: Oversampling")
    dataset3 = ImbalancedPetDataset(root, "trainval.txt", train_transform, cat_frac=0.2)
    train_size = int(0.8 * len(dataset3))
    valid_size = len(dataset3) - train_size
    train_data, valid_data = random_split(dataset3, [train_size, valid_size])
    sampler = getSampler(dataset3, train_data.indices)
    train_load = DataLoader(train_data, batch_size=32, sampler=sampler)
    valid_load = DataLoader(valid_data, batch_size=32, shuffle=False)
    model3 = getModel()
    criterion3 = nn.CrossEntropyLoss()
    optimizer3 = optim.Adam([p for p in model3.parameters() if p.requires_grad], lr)
    train(model3, train_load, valid_load, optimizer3, criterion3, n_epochs, "best_oversample.pth")
    model3.load_state_dict(torch.load("best_oversample.pth"))
    test_acc3 = evaluation(model3, test_load)
    per_cls3, macro_f1_3, weighted_f1_3 = evaluationDetailed(model3, test_load)
    print(f"Test Accuracy = {test_acc3:.2f}%")
    print(f"Macro F1 = {macro_f1_3:.4f}  Weighted F1 = {weighted_f1_3:.4f}")

    print("\nSummary:")
    print(f"Imbalanced (no fix):  Test={test_acc1:.2f}%  Macro F1={macro_f1_1:.4f}  Weighted F1={weighted_f1_1:.4f}")
    print(f"Weighted CE:          Test={test_acc2:.2f}%  Macro F1={macro_f1_2:.4f}  Weighted F1={weighted_f1_2:.4f}")
    print(f"Oversampling:         Test={test_acc3:.2f}%  Macro F1={macro_f1_3:.4f}  Weighted F1={weighted_f1_3:.4f}")

if __name__ == "__main__":
    main()