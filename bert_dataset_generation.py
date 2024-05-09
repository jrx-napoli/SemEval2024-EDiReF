import json
import os

import matplotlib.pyplot as plt
import nltk
import string
import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from nltk.corpus import stopwords
from torch.utils.data import DataLoader
from nltk.stem import PorterStemmer
from transformers import BertTokenizer, BertModel
from bert_data_wrapper import DialogueDatasetWrapperForBert

import pandas as pd

nltk.download('stopwords')

ERC_DATASET_PATHS = {
    "train": "data/erc/MaSaC_train_erc.json",
    "val": "data/erc/MaSaC_val_erc.json",
    "test": "data/erc/MaSaC_test_erc.json"
}

EMOTIONS_ERC = {
    "disgust": 0,
    "neutral": 1,
    "contempt": 2,
    "anger": 3,
    "sadness": 4,
    "joy": 5,
    "fear": 6,
    "surprise": 7
}


# def get_dataloaders(train_dataset_path, val_dataset_path, batch_size):
#     train_dataloader, val_dataloader = _create_dataloader(train_dataset_path, val_dataset_path, batch_size)
#     return train_dataloader, val_dataloader


def _create_dataloader(train_dataset_path, val_dataset_path, BATCH_SIZE):
    with open(train_dataset_path) as f:
        train_data = json.load(f)

    with open(val_dataset_path) as f:
        val_data = json.load(f)

    train_encoded_emotions, train_cleaned_utterances = clean_data(train_data)
    val_encoded_emotions, val_cleaned_utterances = clean_data(val_data)

    # train_padded_sequences = pad_sequence([torch.tensor(seq) for seq in train_cleaned_utterances], batch_first=True,
    #                                       padding_value=0)
    # val_padded_sequences = pad_sequence([torch.tensor(seq) for seq in val_cleaned_utterances], batch_first=True,
    #                                     padding_value=0)
    #
    # assert len(train_padded_sequences) == len(train_encoded_emotions)
    # assert len(val_padded_sequences) == len(val_encoded_emotions)

    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    max_length = find_max_length(train_cleaned_utterances, val_cleaned_utterances)

    training_data = DialogueDatasetWrapperForBert(utterance=train_cleaned_utterances,
                                                  label=train_encoded_emotions,
                                                  tokenizer=tokenizer,
                                                  max_len=max_length)

    validation_data = DialogueDatasetWrapperForBert(utterance=val_cleaned_utterances,
                                                    label=val_encoded_emotions,
                                                    tokenizer=tokenizer,
                                                    max_len=max_length)

    train_loader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=lambda x: x )
    val_loader = DataLoader(validation_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=lambda x: x )

    sample_batch = next(iter(train_loader))

    print(sample_batch['utterance'])
    print(sample_batch['input_id'])
    print(sample_batch['attention_mask'])
    print(sample_batch['label'])



    return train_loader, val_loader


def find_max_length(train_utterances, val_utterances):
    concatenated_list = train_utterances + val_utterances
    max_length = 0
    for sublist in concatenated_list:
        if len(sublist) > max_length:
            max_length = len(sublist)
    return max_length


def clean_data(dataset):
    utterances = []
    emotions = []

    for dialogue in dataset:
        utterances = utterances + dialogue['utterances']
        emotions = emotions + dialogue['emotions']

    stemmer = PorterStemmer()

    # Remove stop words and punctuation, adding stemming
    filtered_utterances = [[word for word in utterance.split() if word.lower() not in stopwords.words('english')] for
                           utterance in utterances]
    filtered_utterances = [[word.strip(string.punctuation) for word in sublist] for sublist in filtered_utterances]
    filtered_utterances = [[stemmer.stem(word) for word in utterance] for utterance in filtered_utterances]

    # Encode emotions
    encoded_emotions = [EMOTIONS_ERC[emotion] for emotion in emotions]
    return encoded_emotions, filtered_utterances


_create_dataloader(ERC_DATASET_PATHS['train'], ERC_DATASET_PATHS['val'], 16)
