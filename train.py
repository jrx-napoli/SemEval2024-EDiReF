import time
import numpy as np
import torch
import torch.nn as nn
import transformers

from torch import optim
from bert_dataset_generation import calculate_class_weights


def train(model, train_dataloader, n_epochs, device):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(n_epochs):
        losses = []
        correctly_predicted = 0
        total_samples = 0
        start = time.time()

        for iteration, batch in enumerate(train_dataloader):
            x, y = batch
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs.squeeze(), y)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                correctly_predicted += (get_correct_sum(outputs, y)).item()
                total_samples += len(y)
                losses.append(loss.item())

        print("Epoch: {}/{}, loss: {}, accuracy: {} %, took: {} s"
              .format(epoch, n_epochs,
                      np.round(np.mean(losses), 3),
                      np.round(correctly_predicted * 100 / total_samples, 3),
                      np.round(time.time() - start), 3))

    return model


def get_correct_sum(y_pred, y_test):
    _, y_pred_tag = torch.max(y_pred, 1)
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    return correct_results_sum


def train_bert(model, train_dataloader, n_epochs, device):
    labels = []
    for batch in train_dataloader:
        labels.extend(batch['label'].numpy())
    class_weights = calculate_class_weights(labels, device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    #optimizer = optim.SGD(model.parameters(), lr=1e-5)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    total_steps = n_epochs * len(train_dataloader)
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer=optimizer,
                                                             num_warmup_steps=0,
                                                             num_training_steps=total_steps)

    model.train()

    for epoch in range(n_epochs):
        train_loss = []
        accurate_predictions = 0
        start = time.time()
        print(f'Training epoch number: {epoch+1}')

        for iter, batch in enumerate(train_dataloader):
            input_ids = batch['input_id'].to(device)
            attention_masks = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()

            #forward
            predictions = model(input_ids, attention_masks)
            loss = criterion(predictions, labels)
            _, pred_classes = torch.max(predictions, dim=1)
            # if epoch == 19:
            #     print(pred_classes)

            #back
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                train_loss.append(loss.item())
                accurate_predictions += torch.sum(pred_classes==labels).cpu().numpy()

        duration = time.time() - start

        print("Epoch: {}/{}, loss: {}, accuracy: {} %, took: {} s"
              .format(epoch+1, n_epochs,
                      np.round(np.mean(train_loss), 3),
                      np.round(accurate_predictions / len(train_dataloader.dataset) * 100, 3),
                      np.round(duration), 3))

    return model