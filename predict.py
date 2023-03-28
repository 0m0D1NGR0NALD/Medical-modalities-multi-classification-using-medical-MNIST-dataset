import torch
from data import test_loader, dataset
from sklearn.metrics import classification_report

device = torch.device("cpu")

from model import model

model.eval()
model.to(device)

y_true = []
y_pred = []

for test_data in test_loader:
     test_images, test_labels = test_data[0].to(device), test_data[1].to(device)
     pred = model(test_images).argmax(dim=1)
