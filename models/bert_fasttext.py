#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：pycharmCode 
@File    ：bert_fasttext.py
@Author  ：lianghu
@Date    ：2022/4/25 17:45 
"""
from tool import Config
from dataclasses import dataclass, field
from torch import nn
from torch.nn import functional as F
from transformers import BertModel


@dataclass
class ModelConfig(Config):
    model_name: str = field(
        default="bert_fasttext"
    )


class Model(nn.Module):
    def __init__(self, config: Config, pretrained_method: str):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_method)
        embedding_dim = self.bert.config.to_dict()['hidden_size']
        self.fc = nn.Linear(embedding_dim, config.output_dim)

    def forward(self, text):
        # last_hidden_state 为BertModel的输出的一个返回值， 大小为(batch_size, sequence_length, hidden_size)
        # BertModel的输出参考：https://huggingface.co/docs/transformers/main/en/model_doc/bert#transformers.BertModel
        embedded = self.bert(text).last_hidden_state

        # pooled = (batch size, embedding_dim)
        pooled = F.avg_pool2d(embedded, (embedded.shape[1], 1)).squeeze(1)
        return self.fc(pooled)
