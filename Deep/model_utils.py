from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import torch.nn as nn


def load_criteria():
    return CrossEntropyLoss()


def load_optimizer(model: nn.Module):
    return Adam(model.parameters(), lr=0.01)