import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision import datasets, models, transforms
from torchvision.models import ResNet18_Weights
from tqdm import tqdm

learning_rate = 0.001
batch_size = 144
num_epochs = 5

data_path = "data/"
data_groups = ["test", "train", "val"]
group_datasets = {}
group_loaders = {}

if torch.cuda.is_available():
    print("CUDA Detected: Using CUDA to train")
    device = torch.device("cuda")
else:
    print("CUDA not Detected: Using CPU to train")
    device = torch.device("cpu")
print()


img_transform = transforms.Compose(
    [
        # Rotate random up to 15 degrees
        transforms.RandomRotation(degrees=15),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)


def show_transformed_image(dataset, index):
    img, _ = dataset[index]
    img_np = np.transpose(img.numpy(), (1, 2, 0))

    plt.imshow(img_np)
    plt.title(f"Class: {dataset.classes[index]}, Index: {index}")
    plt.axis("off")
    plt.show()


for data_group in data_groups:
    print(f"Parsing data group: {data_group}")
    total_path = data_path + data_group
    gp_dataset = datasets.ImageFolder(root=total_path, transform=img_transform)
    group_datasets[data_group] = gp_dataset
    do_shuffle = data_group == "train"
    print(f"Loading data with shuffle={do_shuffle}")
    group_loaders[data_group] = torch.utils.data.DataLoader(
        gp_dataset, batch_size=batch_size, shuffle=do_shuffle
    )

    total_classes = len(gp_dataset.classes)

    print(f"Identified {total_classes} classes:")
    for class_idx in range(total_classes):
        class_name = gp_dataset.classes[class_idx]
        imge_count = len(os.listdir(total_path + "/" + class_name))
        print(f"{class_idx}: {class_name}, {imge_count} images")
    print()

    # show_transformed_image(gp_dataset, 0)

# Load pretrained ResNet-18
resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
# Modify the classifier of ResNet-18
num_ftrs = resnet.fc.in_features
# Modify output to match number of classes
resnet.fc = nn.Linear(num_ftrs, len(gp_dataset.classes))
# move to CUDA if applicable
resnet = resnet.to(device)

# Set optimizer, learning rate, and loss function
optimizer = optim.Adam(resnet.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()


def train_model(model, criterion, optimizer):
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        # Training phase
        model.train()
        with tqdm(
            group_loaders["train"], desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch"
        ) as iter:
            for inputs, labels in iter:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # Log evaluation data
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total_predictions += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()

                # Update status bar
                iter.set_postfix(
                    loss=running_loss / total_predictions,
                    acc=correct_predictions / total_predictions,
                )

        epoch_loss = running_loss / len(group_datasets["train"])
        epoch_accuracy = correct_predictions / total_predictions

        # Validation phase (evaluate on validation set)
        model.eval()
        val_loss, val_accuracy = evaluate_model(model, criterion)

        # Print progress
        print()
        print(
            f"Epoch [{epoch+1}/{num_epochs}], "
            f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_accuracy:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}"
        )
        print()


def evaluate_model(model, criterion):
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    model.eval()
    with torch.no_grad():
        for inputs, labels in group_loaders["val"]:
            inputs, labels = inputs.to(device), labels.to(device)  # Move to CUDA

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(group_datasets["val"])
    epoch_accuracy = correct_predictions / total_predictions

    return epoch_loss, epoch_accuracy


print("Starting training")
print()
train_model(resnet, criterion, optimizer)

# Example usage: evaluate_model(resnet, criterion, test_loader)
