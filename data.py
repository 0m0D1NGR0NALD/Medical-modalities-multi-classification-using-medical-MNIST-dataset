import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import datasets,transforms,models
from torchvision.utils import make_grid

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

data = "Modalities/"

train_transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(root=data,transform=train_transform)

# Split dataset into train and test sets
train_indices,test_indices = train_test_split(list(range(len(dataset.targets))),test_size=0.2,stratify=dataset.targets)
