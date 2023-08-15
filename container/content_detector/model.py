import torch.nn as nn
from transformers import RobertaModel


# Model with extra classification layers on top of RoBERTa
class ROBERTAClassifier(nn.Module):
    def __init__(self, n_classes=2, dropout_rate=0.5, model_path="roberta-base"):
        super(ROBERTAClassifier, self).__init__()
        self.roberta = RobertaModel.from_pretrained(model_path)
        self.drop = nn.Dropout(dropout_rate)
        self.out = nn.Linear(self.roberta.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        x = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        x1 = self.drop(x.pooler_output)
        x_out = self.out(x1)
        return x_out
