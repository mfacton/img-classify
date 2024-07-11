import time
import copy
import glob

import numpy as np
import pandas as pd
#Allows the python interpreter to edit images
from PIL import Image

from tqdm.auto import tqdm
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18, resnet34
#ToDo: why are these the recommeneded models

normal_train_path = 'data/chest_xray/train/NORMAL'
train_normal = glob.glob(normal_train_path + '/*.jpeg')

pneu_train_path = 'data/chest_xray/train/PNEUMONIA'
train_pneu = glob.glob(pneu_train_path + '/*.jpeg')


normal_test_path = 'data/chest_xray/test/NORMAL'
test_normal = glob.glob(normal_test_path + '/*.jpeg')

pneu_test_path = 'data/chest_xray/test/PNEUMONIA'
test_pneu = glob.glob(pneu_test_path + '/*.jpeg')

#confirm length of train
print(len(train_normal))
print(len(train_pneu))
#confirm length of test
print(len(test_normal))
print(len(test_pneu))

train_paths = train_normal + train_pneu
test_paths = test_normal + test_pneu

train_labels = [0] * len(train_normal) + [1] * len(train_pneu)
test_labels = [0] * len(test_normal) + [1] * len(test_pneu)

print(len(train_paths), len(train_labels))
print(len(test_paths), len(test_labels))

from sklearn.model_selection import train_test_split
train_paths, valid_paths, train_labels, valid_labels = train_test_split(train_paths,
                                                                        train_labels,
                                                                        stratify=train_labels,
                                                                        test_size=0.2, random_state=42)
                                                                        
import random

def show_random_images():
  path_random_normal = random.choice(train_normal)
  path_random_pneu = random.choice(train_pneu)

  fig = plt.figure(figsize=(10, 10))

  #Add an Axes to the current figure or retrieve an existing Axes.
  ax1 = plt.subplot(1, 2, 1)
  ax1.imshow(Image.open(path_random_normal).convert("1")) #.convert('RGB')
  ax1.set_title('Normal X-ray')

  ax2 = plt.subplot(1, 2, 2)
  ax2.imshow(Image.open(path_random_pneu).convert("1"))
  ax2.set_title('Pneumonia X-ray')
  
#show_random_images()
#show_random_images()
#show_random_images()
#show_random_images()

class ChestXrayDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        label = self.labels[index]

        image = Image.open(path).convert("1")

        if self.transform:
            image = self.transform(image)

        label = self.labels
        label = torch.tensor(label)

        return image, label
        
class PneuModel(nn.Module):
    def __init__(self, pretrained = True):
        # Calls constructor from parent class nn.Module
        super(PneuModel, self).__init__()
        #mayve consider using resnet50 because
        #pretrained weights might match better
        self.backbone = resnet18(pretrained=pretrained)
        self.fc = nn.Linear(512, 1)

    # Torchvision defines many functions for models
    # https://pytorch.org/vision/main/models.html

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        x = x.view(x.size(0), 512)
        x = self.fc(x)


        return x
        
image_size = (500,500)

train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB
    transforms.RandomRotation(degrees=15),
    transforms.Resize(size=image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Mean and std should match the number of channels
])

test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize(size=image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

train_dataset = ChestXrayDataset(train_paths, train_labels, train_transform)
valid_dataset = ChestXrayDataset(valid_paths, valid_labels, test_transform)

model = PneuModel(pretrained=True)
# https://pytorch.org/docs/stable/generated/torch.nn.Module.html
# if we need to changed pretrained to weights

num_epochs = 5
train_batch_size = 16
valid_batch_size = 16

train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=valid_batch_size, shuffle=False)


dataloaders = {
    "train": train_dataloader,
    "valid": valid_dataloader
}

logging_steps = {
    "train" : len(dataloaders["train"]) // 10,
    "valid" : len(dataloaders["valid"]) // 10
}

dataset_sizes = {
    "train": len(train_dataset),
    "valid": len(valid_dataset)
}

batch_sizes = {
    'train': train_batch_size,
    'valid': valid_batch_size
}

criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr= 3e-3)

model.train()

model.cuda()

def train_model(model, criterion, optimizer, num_epochs = num_epochs, device = "cuda"):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in tqdm(range(num_epochs), leave = False):
        for phase in ["train", "valid"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for i, (inputs, labels) in tqdm(enumerate(dataloaders[phase]),
                                            total = len(dataloaders[phase]),
                                            leave = False):
                print()

                inputs = inputs.to(device)
                labels = labels.to(device)

                labels = torch.argmax(labels, dim=1)

                # if i == 0:
                #     print(f"[{phase}] Labels shape: {labels.shape}")
                #     print(f"[{phase}] Sample labels: {labels[:5]}")

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)

                    num_classes = outputs.shape[1]
                    #print(f"Number of output classes from the model: (I think... I don't know for sure at this point): {num_classes}")

                    if i == 0:
                        print(f"[{phase}] Outputs shape: {outputs.shape}")
                        print(f"[{phase}] Labels shape: {labels.shape}")

                    preds = outputs.sigmoid() > 0.5 #TODO: Activation Function

                    # Check if labels are within the valid range and adjust if needed
                    if labels.max() >= num_classes:
                        #print("Warning: Labels out of bounds. Adjusting labels...")
                        labels = torch.clamp(labels, 0, num_classes - 1) # Clamp labels to be within the valid range

                    loss = criterion(outputs, labels.long())

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)


                if (i % logging_steps[phase] == 0) & (i > 0):
                    avg_loss = running_loss / ((i+1) * dataset_sizes[phase])
                    avg_acc = running_corrects / ((i+1) * batch_sizes[phase])

                    print(f"[{phase}]: {epoch+1} / {num_epochs} | Loss: {avg_loss} | Accuracy: {avg_acc}")


                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]


                print("{} Loss: {:.4f} Acc: {:4f}".format(phase, epoch_loss, epoch_acc))

                if phase == "valid" and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())


                print()

            time_elapsed = time.time() - since
            print(f"training took {time_elapsed} seconds")

            model.load_state_dict(best_model_wts)
            return model
            
model = train_model(model=model, criterion=criterion, optimizer=optimizer, num_epochs = 5)
