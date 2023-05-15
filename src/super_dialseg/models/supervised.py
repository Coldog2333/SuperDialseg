import torch.nn as nn
from transformers import (
    AutoModel,
    AutoConfig,
    BertForTokenClassification,
    RobertaForTokenClassification,
)

from config_file import ROLE2LABEL, DOC2DIAL_DA2LABEL


class Bert(nn.Module):
    def __init__(self, args):
        super(Bert, self).__init__()
        self.args = args
        self.backbone = BertForTokenClassification.from_pretrained(args.backbone)

    def forward(self, inputs):
        outputs = self.backbone(
            inputs['input_ids'],
            inputs['attention_mask'],
            return_dict=True
        )
        logits = outputs['logits']
        return {'logits': logits}


class RobertaMultiTask(nn.Module):
    def __init__(self, args):
        super(RobertaMultiTask, self).__init__()
        self.args = args

        self.backbone_config = AutoConfig.from_pretrained(args.backbone)
        self.backbone = RobertaForTokenClassification.from_pretrained(args.backbone)

        self.da_classifier = nn.Linear(
            in_features=self.backbone_config.hidden_size,
            out_features=len(DOC2DIAL_DA2LABEL),
            bias=True
        )
        self.role_classifier = nn.Linear(
            in_features=self.backbone_config.hidden_size,
            out_features=len(ROLE2LABEL),
            bias=True
        )

        self.dropout = nn.Dropout(p=0.1)

    def forward(self, inputs):
        outputs = self.backbone(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_hidden_states=True,
            return_dict=True
        )
        # segmentation point classification
        logits = outputs['logits']

        last_hidden_state = outputs['hidden_states'][-1]

        hidden_state = self.dropout(last_hidden_state)  # TODO: check now

        # dialogue act classification
        da_logits = self.da_classifier(hidden_state)

        # dialogue role classification
        role_logits = self.role_classifier(hidden_state)

        return {'logits': logits,
                'da_logits': da_logits,
                'role_logits': role_logits}


class Todbert(nn.Module):
    def __init__(self, args):
        super(Todbert, self).__init__()
        self.args = args

        self.model_config = AutoConfig.from_pretrained(args.backbone)
        self.backbone = AutoModel.from_pretrained(args.backbone)

        if self.args.backbone in ['t5-base']:
            self.backbone = self.backbone.encoder
            self.model_config.layer_norm_eps = 1e-05

        self.segmentation_classifier = nn.Sequential(
            nn.Linear(self.model_config.hidden_size, self.model_config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(self.model_config.hidden_size, eps=self.model_config.layer_norm_eps),
            nn.Linear(self.model_config.hidden_size,
                      out_features=2,
                      bias=True)
        )

    def forward(self, inputs):
        outputs = self.backbone(inputs["input_ids"],
                                attention_mask=inputs["attention_mask"],
                                return_dict=True)
        last_hidden_state = outputs['last_hidden_state']
        logits = self.segmentation_classifier(last_hidden_state)
        return {'logits': logits}
