import transformers
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification

class EvidenceScoringModel(torch.nn.Module):
    def __init__(self, args):
        super(EvidenceScoringModel, self).__init__()
        out_dim = 4
        embedding_dim = 768

        self.model = AutoModelForSequenceClassification.from_pretrained("distilroberta-base", num_labels=4)

        for param in self.model.parameters():
            if args.freeze_model:
                param.requires_grad = False
            else:
                param.requires_grad = True

        self.fc = torch.nn.Sequential(
            torch.nn.BatchNorm1d(embedding_dim, eps=1e-05, momentum=0.1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5, inplace=False),
            #torch.nn.MultiheadAttention(embedding_dim, 2)
            torch.nn.Linear(embedding_dim, out_dim))

    def forward(self, input_ids, attention_mask):
        #input_ids = input_ids.squeeze(1)
        # for i in range(len(input_ids)):
        # input_ids = torch.reshape(input_ids, (320, 512))
        # attention_mask = torch.reshape(attention_mask, (320, 512))
        # input_ids = input_ids.long()
        # attention_mask = attention_mask.long()
        hidden_out = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True).hidden_states[-1]
        out = torch.mean(hidden_out, dim=1)
        out = F.normalize(out, dim=-1)
        out = self.fc(out)
        return out