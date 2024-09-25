import torch
import torch.nn as nn
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertEncoder


class CrossAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(CrossAttention, self).__init__()
        self.bert_config = BertConfig()
        self.bert_config.hidden_size = input_dim
        self.bert_config.num_hidden_layers = 1
        self.bert_config.num_attention_heads = 1
        self.bert_config.intermediate_size = hidden_dim
        self.bert_config.hidden_act = 'gelu'
        self.bert_config.output_attentions = True
        self.encoder = BertEncoder(self.bert_config)

    def forward(self, antibody, antigen):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder = self.encoder.to(device)

        antibody_mask = torch.ones(1, antibody.size(1), dtype=torch.bool).to(device)
        antigen_mask = torch.ones(1, antigen.size(1), dtype=torch.bool).to(device)
        attention_mask = torch.cat([antibody_mask, antigen_mask], dim=1).to(device)
        attention_mask = ~attention_mask
        attention_mask = attention_mask.to(dtype=torch.float32)

        input_sequence = torch.cat([antibody, antigen], dim=1).to(device)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        head_mask = [None] * self.bert_config.num_hidden_layers
        encoder_outputs = self.encoder(input_sequence, attention_mask=extended_attention_mask, head_mask=head_mask, output_attentions=True)

        attention_probs = encoder_outputs.attentions[-1]

        antibody_antigen_attention = attention_probs[:, :, :antibody.size(1), antibody.size(1):]

        return antibody_antigen_attention



