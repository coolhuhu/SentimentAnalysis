#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：pycharmCode 
@File    ：bilstm_attention.py
@Author  ：lianghu
@Date    ：2022/3/24 16:47 
"""

import torch
from torch import nn as nn
from torch.nn import functional as F
from dataclasses import dataclass, field
from typing import Optional
from tool import Config


@dataclass
class ModelConfig(Config):
    model_name: str = field(
        default="bilstm_attention"
    )
    lstm_hidden_size: int = field(
        default=128
    )
    lstm_num_layers: int = field(
        default=2
    )
    fc_hidden_size: int = field(
        default=64
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
    """
        使用BiLSTM进行语义学习，使用注意力机制学习BiLSTM输出的权重，最后将BiLSTM的输出加权平均作为句子的向量表示；
    """

    def __init__(self, config: ModelConfig):
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

        self.tanh = nn.Tanh()
        self.w = nn.Parameter(torch.zeros(config.lstm_hidden_size * 2))
        self.fc1 = nn.Linear(config.lstm_hidden_size * 2, config.fc_hidden_size)
        self.fc2 = nn.Linear(config.fc_hidden_size, config.output_dim)

    def forward(self, text):
        # embedded = (batch size, sequence length, embedding_dim) when batch_first = True
        embedded = self.embedding(text)

        # H = (batch size, sequence length, hidden_size * num_direction)
        H, _ = self.lstm(embedded)
        H = self.tanh(H)

        # score = (batch size, sequence length)
        score = torch.matmul(H, self.w)
        # score = (batch size, sequence length, 1)
        score = F.softmax(score, dim=1).unsqueeze(-1)

        # output = (batch size, hidden_size)
        output = torch.sum(H * score, 1)

        output = F.relu(output)
        # output = (batch size, fc_hidden_size)
        output = self.fc1(output)
        # output = (batch size, output_dim)
        output = self.fc2(output)

        return output
