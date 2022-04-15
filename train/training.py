from sklearn.metrics import roc_curve, auc, accuracy_score
import numpy as np
import torch
from train.metrics import AUC as accuracy


def train_epoch(model, optimizer, device, data_loader):
    model.train()
    epoch_loss = 0
    epoch_train_acc = 0
    nb_data = 0
    gpu_mem = 0
    for iter, (batch_feature, batch_labels) in enumerate(data_loader):
        batch_x = batch_feature.to(device)  # num x feat
        batch_labels = batch_labels.to(device)
        optimizer.zero_grad()
        batch_scores = model.forward(batch_x)
        score_lst = []
        labels_lst = []
        for i in range(len(batch_scores)):
            score_value = batch_scores[i]#float(batch_scores[i][1].item())
            lab_value = batch_labels[i]# int(torch.argmax(batch_labels[i]).item())
            score_lst.append(score_value.detach().numpy())
            labels_lst.append(lab_value.detach().numpy())
        score_lst = np.array(score_lst)
        labels_lst = np.array(labels_lst)
        loss = model.loss(batch_scores, batch_labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
        epoch_train_acc += accuracy(score_lst, labels_lst)
    epoch_loss /= (iter + 1)
    epoch_train_acc /= (iter + 1)

    return epoch_loss, epoch_train_acc, optimizer


def evaluate_network(model, device, data_loader):
    model.eval()
    epoch_test_loss = 0
    epoch_test_acc = 0
    nb_data = 0
    with torch.no_grad():
        for iter, (batch_feature, batch_labels) in enumerate(data_loader):
            batch_x = batch_feature.to(device)
            batch_labels = batch_labels.to(device)
            batch_scores = model.forward(batch_x)
            score_lst = []
            labels_lst = []
            for i in range(len(batch_scores)):
                score_value = batch_scores[i]  # float(batch_scores[i][1].item())
                lab_value = batch_labels[i]
                score_lst.append(score_value.detach().numpy())
                labels_lst.append(lab_value.detach().numpy())
            score_lst = np.array(score_lst)
            labels_lst = np.array(labels_lst)
            loss = model.loss(batch_scores, batch_labels)
            epoch_test_loss += loss.detach().item()
            epoch_test_acc += accuracy(score_lst, labels_lst)
        epoch_test_loss /= (iter + 1)
        epoch_test_acc /= (iter + 1)

    return epoch_test_loss, epoch_test_acc
