import torch
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
from Deep.dataloader import GraphsDataset
from Deep.model import load_model
from torch import nn
from typing import NoReturn
from torch_geometric.data import DataLoader
from torchvision.models import vgg16
from Deep.model_utils import load_criteria, load_optimizer
from conf_pack.configuration import default_params
from utils import get_data_path
# Train -> Dataloader -> Pre-process -> configuration
# Train -> utils


def main_train():
    dataset = GraphsDataset(root=get_data_path())
    # model = load_model(num_feat=924, num_classes=2)
    model = load_model(num_feat=7, num_classes=2).double()
    dl = DataLoader(dataset, batch_size=default_params.getint('batch_size'), shuffle=True)
    train_model(model, dl)


def train_model(model: nn.Module, dl: DataLoader) -> NoReturn:
    criteria = load_criteria()
    optimizer = load_optimizer(model)
    with torch.enable_grad():
        for epoch in range(default_params.getint('num_epochs')):
            avg_loss, avg_acc = 0, 0
            print(f'Start epoch number {epoch}')
            for num_batches, (graph, label) in enumerate(dl, start=1):

                print(f'Ran batch {num_batches}')
                optimizer.zero_grad()
                output = model(graph)
                loss = criteria(output, label)
                loss.backward()
                optimizer.step()
                output = F.softmax(output, dim=1)
                _, preds = torch.max(output, dim=1)
                avg_loss += loss.item()
                avg_acc += accuracy_score(y_true=label, y_pred=preds)
                # print(f'avg_acc is {avg_acc} / {num_batches}')
            print(f'The loss the epoch is {avg_loss / num_batches}')
            print(f'The accuracy the epoch is {avg_acc / num_batches}')
                # print(f'The loss is : {loss.item()}')
                # print(f' The accuracy is {accuracy_score(y_true=label, y_pred=preds)}')


if __name__ == '__main__':
    main_train()