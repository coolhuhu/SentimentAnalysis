#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：pycharmCode 
@File    ：bilstm.py
@Author  ：lianghu
@Date    ：2022/4/25 16:08 
"""

import torch
from torch import nn as nn
from dataclasses import dataclass, field
from typing import Optional
from tool import Config


@dataclass
class ModelConfig(Config):
    model_name: str = field(
        default="bilstm"
    )
    lstm_hidden_size: int = field(
        default=128
    )
    lstm_num_layers: int = field(
        default=2
    )
    vocab_size: Optional[int] = field(
        default=None
    )
    embedding_dim: Optional[int] = field(
        default=None
    )
    padding_idx: Optional[int] = field(
        default=None
    )

    def set_embedding_parameters(self, vocab_size, embedding_dim, padding_idx):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx


class Model(nn.Module):
    def __init__(self, config: Config):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=config.vocab_size,
                                      embedding_dim=config.embedding_dim,
                                      padding_idx=config.padding_idx)
        self.lstm = nn.LSTM(input_size=config.embedding_dim,
                            hidden_size=config.lstm_hidden_size,
                            num_layers=config.lstm_num_layers,
                            batch_first=True,
                            dropout=0 if config.lstm_num_layers < 2 else config.dropout,
                            bidirectional=True)
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.lstm_hidden_size * config.lstm_num_layers, config.output_dim)

    def forward(self, text):
        embedded = self.embedding(text)
        output, (hidden, c) = self.lstm(embedded)
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        hidden = self.dropout(hidden)

        return self.fc(hidden)
