"""
Minimal PyTorch training example for hyperparameter sweeps.

Task:
    FashionMNIST image classification

Goal for students:
    Explore the effect of
        - learning rate
        - batch size
        - weight decay
        - dropout
        - learning rate schedules

This script is intentionally simple so you could run it on a CPU.
Typical runtime: ~1-3 minutes.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import wandb


# ------------------------------------------------------------
# 1. Configuration
# ------------------------------------------------------------
# Default hyperparameters.
# When running a W&B sweep these will automatically be overridden.
config = dict(
    batch_size=256,
    learning_rate=1e-3,
    weight_decay=0.0,
    dropout=0.,
    epochs=40,
    filters=[32, 64, 128, 256],
)

wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name / user name).
    entity="nihermann",
    # Set the wandb project where this run will be logged.
    project="param-tuning-demo",
    # Track hyperparameters and run metadata.
    config=config
)
config = wandb.config


# ------------------------------------------------------------
# 2. Device
# ------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ------------------------------------------------------------
# 3. Dataset
# ------------------------------------------------------------
# FashionMNIST images are 28x28 grayscale images of clothing items.

transform = transforms.ToTensor()

dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform,
)

# only select a subset of the classes to make the task easier and overfitting easier to observe.
# subset_mask = (2 <= dataset.targets) & (dataset.targets <= 4)
# dataset.data = dataset.data[subset_mask]
# dataset.targets = dataset.targets[subset_mask]

# To make overfitting easier to observe we intentionally
# train only on a subset of the data.
size = len(dataset.targets) // 10  # use only 10% of the data, which is ~5000 images for CIFAR10
train_dataset, val_dataset, test_dataset = random_split(
    dataset,
    [int(size * 0.6), int(size * 0.2), len(dataset.targets) - int(size * 0.8)],
    generator=torch.Generator().manual_seed(42),
)

train_loader = DataLoader(
    train_dataset,
    batch_size=config.batch_size,
    shuffle=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=config.batch_size,
)


# ------------------------------------------------------------
# 4. Model
# ------------------------------------------------------------
# A small convolutional neural network.
# It is intentionally slightly overparameterized for the small dataset.

def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(),
    )


class SimpleCNN(nn.Module):
    def __init__(self, dropout: float = 0.2) -> None:
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            conv_block(3, config.filters[0]),
            conv_block(config.filters[0], config.filters[1]),
            nn.MaxPool2d(2),
            conv_block(config.filters[1], config.filters[2]),
            conv_block(config.filters[2], config.filters[3]),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(config.filters[-1] * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


model = SimpleCNN(config.dropout).to(device)


# ------------------------------------------------------------
# 5. Loss + Optimizer
# ------------------------------------------------------------
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=config.learning_rate,
    weight_decay=config.weight_decay,
)


# ------------------------------------------------------------
# 6. Optional learning rate scheduler
# ------------------------------------------------------------
# Students can experiment by turning this on/off.

# scheduler = optim.lr_scheduler.CosineAnnealingLR(
#     optimizer,
#     T_max=config.epochs,
# )


# ------------------------------------------------------------
# 7. Training loop
# ------------------------------------------------------------

def train_epoch():
    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    accuracy = correct / total

    return total_loss / len(train_loader), accuracy


# ------------------------------------------------------------
# 8. Validation loop
# ------------------------------------------------------------

def validate():
    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            total_loss += loss.item()

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total

    return total_loss / len(val_loader), accuracy


# ------------------------------------------------------------
# 9. Main training loop
# ------------------------------------------------------------

for epoch in range(config.epochs):

    train_loss, train_acc = train_epoch()
    val_loss, val_acc = validate()

    # scheduler.step()

    # Log results to W&B so we can compare different runs
    wandb.log(
        {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"],
        }
    )

    print(
        f"Epoch {epoch:02d} | "
        f"train_loss={train_loss:.3f} | "
        f"train_acc={train_acc:.3f} | "
        f"val_acc={val_acc:.3f}"
    )


print("Training finished.")