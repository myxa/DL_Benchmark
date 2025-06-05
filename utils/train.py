import torch
from tqdm import tqdm
import numpy as np
import pickle
from sklearn.model_selection import StratifiedKFold
from torchmetrics.classification import BinaryAccuracy


def train_epoch(train_loader, model, criterion, optimizer):
    model.train()
    for x, y in train_loader:
        out = model(x.to(device()))
        loss = criterion(out, y.to(device()))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def eval_epoch(loader, model, criterion):
    model.eval()
    losses = []
    acc = []
    accuracy = BinaryAccuracy()
    #k = 0
    with torch.no_grad():
        for x, y in loader:
            #data = data.to(device())
            out = model(x.to(device()))
            loss = criterion(out, y.to(device()))
            losses.append(loss.item())
            #k +=1
            pred = out.argmax(dim=1)
            acc.append(accuracy(pred.cpu(), y.cpu()))

    return np.mean(losses), np.mean(acc)


def train(model, epochs, train_loader, val_loader, criterion, optimizer, scheduler=None, save_best=False, path_to_save=None):

    history = []
    best_val_loss = 1000
    for epoch in tqdm(range(1, epochs+1)):
        train_epoch(train_loader, model, criterion, optimizer)
        train_loss, train_acc = eval_epoch(train_loader, model, criterion)
        val_loss, test_acc = eval_epoch(val_loader, model, criterion)
        if scheduler is not None:
            scheduler.step()

        if save_best:
            if best_val_loss > val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), path_to_save)


        print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Test Loss {val_loss:.4f}, '
              f'Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
        print()

        history.append((train_loss, val_loss, train_acc, test_acc))

    if save_best:
        model.load_state_dict(torch.load(path_to_save, map_location=device()))

    return history

def device():
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')