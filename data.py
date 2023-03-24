import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import datasets,transforms,models
from torchvision.utils import make_grid

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Initializing dataset path
data = "Modalities/"

# Composing augmentation for train set
train_transform = transforms.Compose([
    transforms.RandomRotation(10), # Rotate +|- 10 degrees
    transforms.RandomHorizontalFlip(), # Reverse 50% of images
    transforms.Resize(224), # Resize shortest side to 224 pixels
    transforms.ToTensor(), # Crop longest size to 224 pixels at center
    transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225])
])

# Creating dataset
dataset = datasets.ImageFolder(root=data,transform=train_transform)

# Split dataset into train and test sets
train_indices,test_indices = train_test_split(list(range(len(dataset.targets))),test_size=0.2,stratify=dataset.targets)

train_data = torch.utils.data.Subset(dataset,train_indices)
test_data = torch.utils.data.Subset(dataset,test_indices)

# Datasets to dataloader
train_loader = DataLoader(train_data,batch_size=10,shuffle=True)
test_loader = DataLoader(test_data,batch_size=10)
