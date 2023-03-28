import torch
from data import test_loader, dataset
from sklearn.metrics import classification_report

device = torch.device("cpu")

from model import model
