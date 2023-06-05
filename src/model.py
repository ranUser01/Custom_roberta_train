
import torch
import torch.nn as nn
from torch.nn import functional as F
# from transformers import BertModel, BertTokenizer
from transformers import AutoConfig
from transformers import AutoModelWithLMHead
from src.utils import load_embedding

import logging
logger = logging.getLogger()

class BertTagger(nn.Module):
    def __init__(self, params):
        super(BertTagger, self).__init__()
        self.num_tag = params.num_tag
        self.hidden_dim = params.hidden_dim
        config = AutoConfig.from_pretrained(params.model_name)
        config.output_hidden_states = True
        # self.bert = BertModel.from_pretrained("bert-base-cased")
        self.model = AutoModelWithLMHead.from_pretrained(params.model_name, config=config)
        if params.ckpt != "":
            logger.info("Reloading model from %s" % params.ckpt)
            model_ckpt = torch.load(params.ckpt)
            self.model.load_state_dict(model_ckpt)

        self.linear = nn.Linear(self.hidden_dim, self.num_tag)

    def forward(self, X):
        outputs = self.model(X) # a tuple ((bsz,seq_len,hidden_dim), (bsz, hidden_dim))
        outputs = outputs[1][-1] # (bsz, seq_len, hidden_dim)
        
        prediction = self.linear(outputs)

        return prediction
