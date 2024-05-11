import numpy as np
import torch.nn as nn
import torch

from sklearn.metrics import precision_score, recall_score, f1_score


def evaluate(model, test_dataloader, device):
    model.eval()
    correctly_predicted = 0
    total_samples = 0

    for iteration, batch in enumerate(test_dataloader):
        x, y = batch
        x = x.to(device)
        y = y.to(device)

        outputs = model(x)

        correctly_predicted += (get_correct_sum(outputs, y)).item()
        total_samples += len(y)

    print("Test accuracy: {} %".format(np.round(correctly_predicted * 100 / total_samples, 3)))


def get_correct_sum(y_pred, y_test):
    _, y_pred_tag = torch.max(y_pred, 1)
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    return correct_results_sum


def evaluate_bert(model, data_loader, device):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    eval_loss = []
    accurate_predictions = 0
    all_predictions = []
    all_labels = []

    for d in data_loader:
        input_ids = d['input_id'].to(device)
        attention_masks = d['attention_mask'].to(device)
        labels = d['label'].to(device)

        predictions = model(input_ids, attention_masks)
        loss = criterion(predictions, labels)
        _, pred_classes = torch.max(predictions, dim=1)

        eval_loss.append(loss.item())
        accurate_predictions += torch.sum(pred_classes == labels).cpu().numpy()
        all_predictions.extend(pred_classes.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')

    print("Test accuracy: {} %, loss: {}, precision: {}, recall: {}, F1-score: {}"
          .format(np.round(accurate_predictions * 100 / len(data_loader.dataset), 3),
                  np.mean(eval_loss),
                  np.round(precision, 3),
                  np.round(recall, 3),
                  np.round(f1, 3)))
