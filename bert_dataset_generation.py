import json
import nltk
import string
import torch
import numpy as np

from torch.nn.utils.rnn import pad_sequence
from nltk.corpus import stopwords
from torch.utils.data import DataLoader
from nltk.stem import PorterStemmer
from transformers import BertTokenizer
from bert_data_wrapper import DialogueDatasetWrapperForBert
from sklearn.utils.class_weight import compute_class_weight


if not nltk.corpus.stopwords.words('english'):
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


def get_dataloaders(args):
    train_dataloader, val_dataloader = _create_dataloader(ERC_DATASET_PATHS['train'], ERC_DATASET_PATHS['val'],
                                                          args.batch_size)
    return train_dataloader, val_dataloader


def _create_dataloader(train_dataset_path, val_dataset_path, BATCH_SIZE):
    with open(train_dataset_path) as f:
        train_data = json.load(f)

    with open(val_dataset_path) as f:
        val_data = json.load(f)

    train_encoded_emotions, train_cleaned_utterances = clean_data(train_data)
    val_encoded_emotions, val_cleaned_utterances = clean_data(val_data)

    assert len(train_cleaned_utterances) == len(train_encoded_emotions)
    assert len(val_cleaned_utterances) == len(val_encoded_emotions)

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

    train_loader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=12,
                              collate_fn=custom_collate_fn)
    val_loader = DataLoader(validation_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=12,
                            collate_fn=custom_collate_fn)

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


def custom_collate_fn(data):
    # Custom collate function to pad data dynamically during batch creation
    utterances = [d['utterance'] for d in data]
    #inputs = [torch.tensor(d['input_id']) for d in data]
    inputs = [torch.tensor(d['input_id']).clone().detach() if not isinstance(d['input_id'], torch.Tensor) else d['input_id'].clone().detach() for d in data]

    #attention_masks = [torch.tensor(d['attention_mask']) for d in data]
    attention_masks = [torch.tensor(d['attention_mask']).clone().detach() if not isinstance(d['attention_mask'], torch.Tensor) else d['attention_mask'].clone().detach() for d in data]
    labels = [d['label'] for d in data]

    inputs = pad_sequence(inputs, batch_first=True)
    attention_masks = pad_sequence(attention_masks, batch_first=True)
    labels = torch.tensor(labels)

    return {
        'utterance': utterances,
        'input_id': inputs,
        'attention_mask': attention_masks,
        'label': labels
    }


def calculate_class_weights(labels, device):
    # class_counts = np.bincount(labels)
    # total_samples = len(labels)
    # class_weights = [total_samples / count for count in class_counts]
    class_wts = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
    class_weights = torch.tensor(class_wts, dtype=torch.float)
    class_weights = class_weights.to(device)
    print(class_weights)
    return class_weights
