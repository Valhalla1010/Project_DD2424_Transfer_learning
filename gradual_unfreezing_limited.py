import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import random_split, DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from gradual_unfreezing import PetBreedDataset, getModel, evaluation, unfreeze_layer

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")


def Transforms_augmented():
    Train_transform = transforms.Compose([
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
    return Train_transform, Test_transform


def Transforms_basic():
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


def get_fraction_subset(train_data, fraction):
    """Stratified subset of train_data at the given fraction."""
    indices = list(range(len(train_data)))
    labels = [train_data.dataset.samples[i][1] for i in train_data.indices]

    if fraction >= 1.0:
        return train_data

    # For very small fractions (e.g. 1%) ensure at least one sample per class
    n_samples = max(int(fraction * len(indices)), len(set(labels)))
    selected, _ = train_test_split(
        indices,
        train_size=n_samples,
        stratify=labels,
        random_state=42
    )
    return Subset(train_data, selected)


def train_gradual(model, train_loader, valid_loader, optimizer, criterion,
                  n_epochs, unfreeze_schedule, unfreeze_lr):
    since = time.time()
    best_acc = 0
    best_path = "best_gradual_limited.pth"
    train_history = []
    valid_history = []

    for epoch in range(n_epochs):
        # Gradual unfreezing
        if epoch in unfreeze_schedule:
            layer_name = unfreeze_schedule[epoch]
            unfreeze_layer(model, layer_name, optimizer, unfreeze_lr)

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
        train_history.append(train_acc)
        valid_history.append(valid_acc)

        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save(model.state_dict(), best_path)

        unfrozen = sorted({n.split(".")[0] for n, p in model.named_parameters() if p.requires_grad})
        print(f"Epoch [{epoch + 1}/{n_epochs}]"
              f"\nUnfrozen layers : {unfrozen}"
              f"\nTrain Loss      = {epoch_loss:.4f}"
              f"\nTrain Accuracy  = {train_acc:.2f}%"
              f"\nValid Accuracy  = {valid_acc:.2f}%\n")

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s  :-)")
    print(f"Best Validation Accuracy = {best_acc:.2f}%")
    return train_history, valid_history


def plot_learning_curves(train_acc, valid_acc, title, filename):
    epochs = range(1, len(train_acc) + 1)
    plt.figure()
    plt.plot(epochs, train_acc, label="Train Accuracy")
    plt.plot(epochs, valid_acc, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    print(f"Saved plot: {filename}")


def plot_fraction_results(fractions, acc_no_aug, acc_aug, filename):
    plt.figure()
    plt.plot(fractions, acc_no_aug, marker="o", label="No Aug / No L2")
    plt.plot(fractions, acc_aug,    marker="s", label="Aug + L2")
    plt.xlabel("Training Data Fraction")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Gradual Unfreezing: Test Accuracy vs Data Fraction")
    plt.xscale("log")
    plt.xticks(fractions, ["1%", "10%", "100%"])
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    print(f"Saved plot: {filename}")


def run_experiment(fraction, augment, l2, root, valid_data, test_set):
    """Run one experiment for a given fraction, augmentation and L2 setting."""
    label = f"frac{int(fraction*100)}_{'aug' if augment else 'noaug'}_{'l2' if l2 else 'nol2'}"
    print(f"\n{'='*60}")
    print(f"Fraction={fraction*100:.0f}%  Augmentation={augment}  L2={l2}")
    print(f"{'='*60}")

    if augment:
        train_transform, test_transform = Transforms_augmented()
    else:
        train_transform, test_transform = Transforms_basic()

    # Reload dataset with correct transform
    dataset = PetBreedDataset(root, "trainval.txt", train_transform)
    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size
    train_data, _ = torch.utils.data.random_split(dataset, [train_size, valid_size],
                                                   generator=torch.Generator().manual_seed(42))

    train_subset = get_fraction_subset(train_data, fraction)

    train_load = DataLoader(train_subset, batch_size=32, shuffle=True)
    valid_load = DataLoader(valid_data,   batch_size=32, shuffle=False)
    test_load  = DataLoader(test_set,     batch_size=32, shuffle=False)

    model = getModel()
    criterion = nn.CrossEntropyLoss()
    lr = 0.001
    unfreeze_lr = 0.0001
    weight_decay = 1e-4 if l2 else 0.0

    optimizer = optim.Adam(model.fc.parameters(), lr=lr, weight_decay=weight_decay)

    unfreeze_schedule = {
        3: "layer4",
        6: "layer3",
        9: "layer2",
        12: "layer1",
    }
    n_epochs = 15

    train_acc, valid_acc = train_gradual(
        model, train_load, valid_load, optimizer, criterion,
        n_epochs, unfreeze_schedule, unfreeze_lr
    )

    model.load_state_dict(torch.load("best_gradual_limited.pth"))
    test_acc = evaluation(model, test_load)
    print(f"\nFraction={fraction*100:.0f}%  Test Accuracy = {test_acc:.2f}%")

    plot_learning_curves(
        train_acc, valid_acc,
        title=f"Gradual Unfreezing {fraction*100:.0f}% {'Aug+L2' if augment else 'No Aug'}",
        filename=f"gradual_limited_{label}.png"
    )

    return test_acc


def main():
    root = "Datasets"
    _, test_transform = Transforms_basic()

    # Shared validation and test sets (no augmentation)
    dataset_base = PetBreedDataset(root, "trainval.txt", test_transform)
    test_set = PetBreedDataset(root, "test.txt", test_transform)
    train_size = int(0.8 * len(dataset_base))
    valid_size = len(dataset_base) - train_size
    _, valid_data = torch.utils.data.random_split(
        dataset_base, [train_size, valid_size],
        generator=torch.Generator().manual_seed(42)
    )

    fractions = [0.01, 0.1, 1.0]
    results_no_aug = []
    results_aug    = []

    for frac in fractions:
        acc = run_experiment(frac, augment=False, l2=False,
                             root=root, valid_data=valid_data, test_set=test_set)
        results_no_aug.append(acc)

        acc = run_experiment(frac, augment=True, l2=True,
                             root=root, valid_data=valid_data, test_set=test_set)
        results_aug.append(acc)

    # Summary table
    print("\n--- Summary ---")
    print(f"{'Fraction':<10} {'No Aug/L2':>12} {'Aug+L2':>10}")
    for frac, a, b in zip(fractions, results_no_aug, results_aug):
        print(f"{frac*100:.0f}%{'':<7} {a:>10.2f}%  {b:>8.2f}%")

    plot_fraction_results(
        [f * 100 for f in fractions],
        results_no_aug,
        results_aug,
        filename="gradual_limited_fraction_results.png"
    )


if __name__ == "__main__":
    main()
