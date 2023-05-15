import torch.nn as nn
from transformers import BertForNextSentencePrediction


class CoherenceScoringModel(nn.Module):
    def __init__(self, backbone='bert-base-uncased'):
        super(CoherenceScoringModel, self).__init__()
        self.bert = BertForNextSentencePrediction.from_pretrained(
            backbone,
            num_labels=2,
            output_attentions=False,
            output_hidden_states=True
        )
        self.coherence_prediction_decoder = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.1),
            nn.Linear(768, 2)
        )

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        return_dict=True
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=return_dict
        )
        hidden_states = outputs['hidden_states'][-1]
        cls_hidden_states = hidden_states[:, 0, :]
        # cls_hidden_states = outputs['last_hidden_state'][:, 0, :]
        logits = self.coherence_prediction_decoder(cls_hidden_states)
        # coherence_scores = torch.sigmoid(logits)
        return {'logits': logits}
