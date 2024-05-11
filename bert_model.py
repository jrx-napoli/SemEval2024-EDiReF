from transformers import BertModel
import torch.nn as nn


class BERT(nn.Module):
    def __init__(self, num_classes=8):
        super(BERT, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-multilingual-cased")
        self.dropout = nn.Dropout(p=0.3)
        self.linear = nn.Linear(self.bert.config.hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_id, attention_mask):
        temp = self.bert(input_id, attention_mask)
        pooled_output = temp[1]
        out = self.dropout(pooled_output)
        out = self.linear(out)
        return out
