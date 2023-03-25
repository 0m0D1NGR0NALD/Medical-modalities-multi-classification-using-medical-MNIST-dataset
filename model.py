from data import dataset,train_loader,test_loader
import torch.nn as nn
import torch.functional as F

class_names = dataset.classes

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,6,3,1)
        self.conv2 = nn.Conv2d(6,16,3,1)
        self.fc1 = nn.Linear(16*54*54,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,20)
        self.fc4 = nn.Linear(20,len(class_names))
