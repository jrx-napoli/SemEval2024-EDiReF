import torch
from torch.utils.data import Dataset


class DialogueDatasetWrapperForBert(Dataset):
    def __init__(self, utterance, label, tokenizer, max_len):
        self.utterance = utterance
        self.label = label
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.utterance)

    def __getitem__(self, index):
        utterance = " ".join(self.utterance[index])
        label = self.label[index]
        encoding = self.tokenizer.encode_plus(
            utterance,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            return_token_type_ids=False,
            padding='longest',
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {'utterance': utterance,
                'input_id': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'label': torch.tensor(label, dtype=torch.long)
                }
