#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：pycharmCode 
@File    ：util1.py
@Author  ：lianghu
@Date    ：2022/3/24 20:21 
"""

from torchtext.legacy import data
import logging
from dataclasses import dataclass, field
import torch
from torchtext import vocab
# from models import bilstm_attention
from transformers import BertTokenizer


class Logger:
    """
        将日志信息打印到控制台和记录到文件的操作封装成一个类
    """

    def __init__(self, name: str, console_handler_level: str = logging.DEBUG,
                 fmt: str = '%(asctime)s: %(levelname)s: %(name)s: %(filename)s: %(message)s'):
        """
            默认会添加一个等级为 'DEBUG' 的 Handler 对象到 Logger 对象
        :param name: handler 的名称
        :param console_handler_level: 设置 StreamHandler 的等级
        :param fmt: 日志消息的显示格式
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self.fmt = logging.Formatter(fmt)
        self.set_console_handler(console_handler_level)

    def set_console_handler(self, console_handler_level: str = logging.DEBUG) -> None:
        """
            添加一个StreamHandler
        :param console_handler_level: StreamHandler等级
        :return:
        """
        ch = logging.StreamHandler()
        ch.setLevel(console_handler_level)
        ch.setFormatter(self.fmt)
        self.logger.addHandler(ch)

    def set_file_handler(self, filename: str, mode: str = 'a', file_handler_level: str = logging.INFO) -> None:
        """
            添加一个 FileHandler
        :param filename: 日志保存的文件名
        :param mode: 写文件模式，默认为 a
        :param file_handler_level: FileHandler等级
        :return:
        """
        fh = logging.FileHandler(filename, mode=mode, encoding='utf-8')
        fh.setLevel(file_handler_level)
        fh.setFormatter(self.fmt)
        self.logger.addHandler(fh)

    def debug(self, msg):
        self.logger.debug(msg)

    def info(self, msg):
        self.logger.info(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg)

    def critical(self, msg):
        self.logger.critical(msg)


@dataclass
class Config:
    """
        抽象出来的公共配置类

        Args:
            data_path: str  数据集存放路径，对应TabularDataset的参数path
            train_filename: str 训练集文件名，对应TabularDataset的参数train
            valid_filename: str 验证集文件名，对应TabularDataset的参数validation
            test_filename: str  测试集文件名，对应TabularDataset的参数test
            output_dim: int 标签种数
            dropout: float  Dropout
            lr: float   学习率
            batch_size: int
            num_epochs: int 训练轮数
            device: torch.device 是否使用GPU进行训练
            state_dict: str checkpoint保存路径
    """
    data_path: str = field(
        default=r"/workspace/pycharmCode/ICDD/Sentiment_Analysis_Reviews/data"
    )
    train_filename: str = field(
        default=r'train_data.csv'
    )
    valid_filename: str = field(
        default=r'valid_data.csv'
    )
    test_filename: str = field(
        default=r'test_data.csv'
    )
    output_dim: int = field(
        default=3
    )
    dropout: float = field(
        default=0.25
    )
    lr: float = field(
        default=0.001
    )
    batch_size: int = field(
        default=8
    )
    num_epochs: int = field(
        default=10
    )
    device: torch.device = field(
        default=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    )
    state_dict: str = field(
        default=r'/workspace/pycharmCode/ICDD/Sentiment_Analysis_Reviews/state_dict'
    )


def load_dataset(config: Config, fields):
    """
    从本地加载数据集
    :param config:
    :param fields:
    :return:
    """
    train_Dataset, val_Dataset, test_Dataset = data.TabularDataset.splits(
        path=config.data_path,
        format='csv',
        train=config.train_filename,
        validation=config.valid_filename,
        test=config.test_filename,
        skip_header=True,
        fields=fields)

    return train_Dataset, val_Dataset, test_Dataset


def data_iterator(config: Config, train, valid, test):
    """
    为数据集创建迭代器
    :param config: ModelConfig
    :param train: Dataset
    :param valid:
    :param test:
    :return:
    """
    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (train, valid, test),
        batch_size=config.batch_size,
        sort=False,
        device=config.device)

    return train_iterator, valid_iterator, test_iterator


def load_word2vec_embeddings(module, config, train_Dataset, text, label,
                             name: str = 'word2vec_cellphone_reviews.txt',
                             cache: str = '/workspace/pycharmCode/ICDD/Sentiment_Analysis_Reviews/data'):
    """
    使用word2vec作为词向量
    :param config:
    :param train_Dataset:
    :param text: data.Field
    :param label: data.LabelField
    :param name:
    :param cache:
    :return: model
    """
    word2vec_embeddings = vocab.Vectors(name=name,
                                        cache=cache,
                                        unk_init=torch.Tensor.normal_)

    text.build_vocab(train_Dataset, vectors=word2vec_embeddings)
    label.build_vocab(train_Dataset)

    padding_idx = text.vocab.stoi[text.pad_token]
    unk_idx = text.vocab.stoi[text.unk_token]
    vocab_size = len(text.vocab)
    embedding_dim = word2vec_embeddings.dim

    config.set_embedding_parameters(vocab_size, embedding_dim, padding_idx)

    model = module.Model(config)

    model.embedding.weight.data.copy_(text.vocab.vectors)
    model.embedding.weight.data[padding_idx] = torch.zeros(embedding_dim)
    model.embedding.weight.data[unk_idx] = torch.zeros(embedding_dim)

    return model


def load_bert_embeddings(pretrained_method: str):
    """
    使用BERT预训练模型作为词向量
    :param pretrained_method: BERT预训练模型，从huggingface上加载
    :return:
    """
    tokenizer = BertTokenizer.from_pretrained(pretrained_method)
    init_token = tokenizer.cls_token_id
    eos_token = tokenizer.sep_token_id
    pad_token = tokenizer.pad_token_id
    unk_token = tokenizer.unk_token_id
    max_input_length = tokenizer.max_model_input_sizes[pretrained_method]

    def fasttext_tokenize(sentence):
        x = tokenizer.tokenize(sentence)
        n_grams = set(zip(*[x[i:] for i in range(2)]))
        for n_gram in n_grams:
            x.append(''.join(n_gram))
        tokens = x[: max_input_length - 2]
        return tokens

    TEXT = data.Field(batch_first=True,
                      use_vocab=False,
                      tokenize=fasttext_tokenize if pretrained_method == 'bert_fasttext' else tokenizer.tokenize,
                      preprocessing=tokenizer.convert_tokens_to_ids,
                      init_token=init_token,
                      eos_token=eos_token,
                      pad_token=pad_token,
                      unk_token=unk_token)

    LABEL = data.LabelField()

    fields = [('label', LABEL), ('comment_processed', TEXT)]

    return fields
