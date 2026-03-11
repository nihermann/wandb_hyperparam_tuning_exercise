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
    batch_size=64,
    learning_rate=1e-4,
    weight_decay=0.0,
    dropout=0.,
    epochs=20,
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


# ------------------------------------------------------------
# 3. Dataset
# ------------------------------------------------------------
# FashionMNIST images are 28x28 grayscale images of clothing items.

transform = transforms.ToTensor()

dataset = datasets.FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform,
)

# To make overfitting easier to observe we intentionally
# train only on a subset of the data.
train_size = 20000
val_size = 4000
train_dataset, val_dataset, _ = random_split(
    dataset,
    [train_size, val_size, len(dataset) - train_size - val_size],
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

class SimpleCNN(nn.Module):
    def __init__(self, dropout: float = 0.2) -> None:
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),  # 28x28 -> 28x28
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), # 28x28 -> 28x28
            nn.ReLU(),
            nn.MaxPool2d(2),                 # -> 14x14

            nn.Conv2d(64, 128, 3, padding=1), # -> 14x14
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1), # -> 14x14
            nn.ReLU(),
            nn.MaxPool2d(2),                 # -> 7x7
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
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