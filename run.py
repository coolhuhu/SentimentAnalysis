#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：pycharmCode 
@File    ：run.py
@Author  ：lianghu
@Date    ：2022/3/25 9:59 
"""

from torchtext.legacy import data
from train_eval import model_train
from torch import nn as nn
import random
import numpy as np
from torch import optim
import torch
import tool
import argparse
from importlib import import_module

parser = argparse.ArgumentParser(description="Sentiment Analysis of JD Mobile Reviews")
parser.add_argument("--model_name", default="bilstm_attention", type=str,
                    choices=['bilstm_attention', 'bilstm_emb_att', 'bilstm', 'bert_fasttext'],
                    help="choose a model from: bilstm_attention, bilstm_emb_att, bilstm, bert_fasttext")
parser.add_argument("--pretrained_method", default="word2vec", type=str, help="pretrained method")
args = parser.parse_args()

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    pretrained_methods = ['bert-base-chinese', 'hfl/chinese-roberta-wwm-ext-large']
    model_name = args.model_name
    pretrained_method = args.pretrained_method

    module = import_module('models.' + model_name)

    config = module.ModelConfig()

    if pretrained_method == "word2vec":
        TEXT = data.Field(batch_first=True)
        LABEL = data.LabelField()
        fields = [('label', LABEL), ('comment_processed', TEXT)]
        train_Dataset, val_Dataset, test_Dataset = tool.load_dataset(config, fields)
        model = tool.load_word2vec_embeddings(module, config, train_Dataset, TEXT, LABEL)
    elif pretrained_method in pretrained_methods:
        fields = tool.load_bert_embeddings(pretrained_method)
        train_Dataset, val_Dataset, test_Dataset = tool.load_dataset(config, fields)
        fields[0][1].build_vocab(train_Dataset)
        model = module.Model(config, pretrained_method)

    train_iterator, valid_iterator, test_iterator = tool.data_iterator(config, train_Dataset, val_Dataset,
                                                                        test_Dataset)

    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    model = model.to(config.device)
    criterion = criterion.to(config.device)

    model_train(config, model, train_iterator, valid_iterator, optimizer, criterion)
