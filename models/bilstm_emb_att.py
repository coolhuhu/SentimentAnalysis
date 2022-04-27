#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：pycharmCode 
@File    ：bilstm_emb_att.py
@Author  ：lianghu
@Date    ：2022/3/26 21:53 
"""

import torch
from torch import nn
from torch.nn import functional as F
from typing import Optional
from dataclasses import dataclass, field
from tool import Config


@dataclass
class ModelConfig(Config):
    model_name: str = field(
        default='bilstm_emb_att'
    )
    W_fc_hidden_size: int = field(
        default=256
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

        self.avg = nn.AdaptiveAvgPool1d(1)

        self.W_fc = nn.Linear(config.embedding_dim, config.W_fc_hidden_size)

        self.tanh = nn.Tanh()

        self.A = nn.Parameter(torch.zeros(config.W_fc_hidden_size))

        self.O_fc = nn.Linear(config.lstm_hidden_size * 2 + config.W_fc_hidden_size, config.output_dim)

    def forward(self, text):
        # embedded = (batch size, sequence length, embedding_dim) when batch_first = True
        embedded = self.embedding(text)

        # H = (batch size, sequence length, hidden_size * num_direction)
        H, _ = self.lstm(embedded)
        H = self.tanh(H)

        # 对BiLSTM单元的输出取平均作为语义学习层输出
        # output_lstm = (batch size, hidden_size * num_direction)
        output_lstm = self.avg(H.permute(0, 2, 1)).squeeze(-1)

        # V = (batch size, sequence length, W_fc_hidden_size)
        V = self.W_fc(embedded)

        # score = (batch size, sequence length)
        score = torch.matmul(V, self.A)
        # score = (batch size, sequence length, 1)
        score = F.softmax(score, dim=1).unsqueeze(-1)

        # output_att = (batch size, W_fc_hidden_size)
        output_att = torch.sum(embedded * score, dim=1)

        # output = (batch size, hidden_size * num_direction + W_fc_hidden_size)
        output = torch.cat((output_lstm, output_att), dim=1)

        return self.O_fc(output)

# if __name__ == "__main__":
#     print(os.path.dirname(sys.path[0]))
#     print(sys.path[0])
