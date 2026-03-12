"""
Minimal PyTorch training example for hyperparameter sweeps.

Task:
    CIFAR-10 image classification with an MLP

Goal for students:
    1. Observe overfitting with the default (large) MLP
    2. Use hyperparameter sweeps to find a configuration that generalizes:
        - learning rate
        - batch size
        - weight decay
        - dropout
        - hidden layer sizes
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
    learning_rate=1e-3,
    weight_decay=0.0,
    dropout=0.0,
    epochs=100,
    num_hidden_layers=3,  # number of hidden layers
    hidden_dim=512,       # units per hidden layer (intentionally large → easy to overfit)
)

username = "<your-user-name>"
wandb_project = "<your-project-name>"

if username == "<your-user-name>" or wandb_project == "<your-project-name>":
    raise ValueError("Please set your W&B username and project name in the config.")

wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name / user name).
    entity=username,
    # Set the wandb project where this run will be logged.
    project=wandb_project,
    # Track hyperparameters and run metadata.
    config=config,
    name="MLP_BASELINE"  # Name of this run; remove it to get randomly generated names
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
# CIFAR-10 images are 32x32 RGB images of objects (10 classes).

transform = transforms.ToTensor()

dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform,
)

# To make overfitting easier to observe we intentionally
# train only on a subset of the data.
size = len(dataset.targets) // 4  # use only 25% of the data, which is ~1250 images for CIFAR10
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
# A simple MLP (multi-layer perceptron).
# It is intentionally overparameterized for the small dataset so that
# overfitting is easy to observe.


INPUT_DIM = 32 * 32 * 3  # CIFAR-10 images flattened
NUM_CLASSES = 10


class SimpleMLP(nn.Module):
    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        num_hidden_layers: int = 3,
        hidden_dim: int = 512,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = [nn.Flatten()]
        prev = input_dim
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(prev, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = hidden_dim
        layers.append(nn.Linear(prev, num_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


model = SimpleMLP(
    num_hidden_layers=config.num_hidden_layers,
    hidden_dim=config.hidden_dim,
    dropout=config.dropout,
).to(device)


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

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=config.epochs,
)


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

    scheduler.step()

    # Log results to W&B so we can compare different runs
    wandb.log({
        "epoch": epoch,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "lr": optimizer.param_groups[0]["lr"],
    })

    print(
        f"Epoch {epoch:02d} | "
        f"train_loss={train_loss:.3f} | "
        f"val_loss={val_loss:.3f} | "
        f"train_acc={train_acc:.3f} | "
        f"val_acc={val_acc:.3f}"
    )


print("Training finished.")