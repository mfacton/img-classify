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

batch_size = 144

model_file = "resnet18_best.pth"

save_path = "models/"
data_path = "data/"

data_groups = ["test", "train", "val"]
group_datasets = {}
group_loaders = {}

if torch.cuda.is_available():
    print("CUDA Detected: Using CUDA to run")
    device = torch.device("cuda")
else:
    print("CUDA not Detected: Using CPU to run")
    device = torch.device("cpu")
print()


img_transform = transforms.Compose(
    [
        # Rotate random up to 15 degrees
        transforms.RandomRotation(degrees=15),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # Normalize to standard resnet input
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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

resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
# Modify the classifier of ResNet-18
num_ftrs = resnet.fc.in_features
# Modify output to match number of classes
resnet.fc = nn.Linear(num_ftrs, len(gp_dataset.classes))
# Load model
resnet.load_state_dict(torch.load(save_path + model_file))
# move to CUDA if applicable
resnet = resnet.to(device)

criterion = nn.CrossEntropyLoss()


def evaluate_model(model, criterion, group_name):
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    model.eval()
    with torch.no_grad():
        for inputs, labels in group_loaders[group_name]:
            inputs, labels = inputs.to(device), labels.to(device)  # Move to CUDA

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    val_loss = running_loss / len(group_datasets[group_name])
    val_accuracy = correct_predictions / total_predictions

    print(f"Eval Loss: {val_loss:.4f}, Eval Acc: {val_accuracy:.4f}")


print("Starting evaluation")
evaluate_model(resnet, criterion, "val")
