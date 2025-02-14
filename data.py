import torch
from torch.utils.data import Dataset
from collections import Counter
import pandas as pd
from config import Config

# 分词词典类
class Vocab:
    def __init__(self, tokens, max_size=None):
        self.tokens = tokens
        self.token_to_idx = {token: idx for idx, token in enumerate(tokens)}
        self.idx_to_token = {idx: token for idx, token in enumerate(tokens)}
        self.pad_idx = self.token_to_idx["<pad>"]
        self.unk_idx = self.token_to_idx["<unk>"]

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, token):
        """获取token对应的index，未知词返回<unk>的index"""
        return self.token_to_idx.get(token, self.unk_idx)

    def lookup_indices(self, tokens):
        return [self[token] for token in tokens]

    def lookup_tokens(self, indices):
        return [self.idx_to_token[idx] for idx in indices]

    @staticmethod
    def build_vocab(texts, max_size=None, min_freq=1):
        token_counter = Counter()
        for text in texts:
            token_counter.update(text.split())

        # 处理特殊标记
        special_tokens = ["<pad>", "<unk>", "<start>", "<end>"]
        tokens = [token for token, freq in token_counter.most_common(max_size)
                 if freq >= min_freq and token not in special_tokens]
        # 保证特殊标记在最前面
        tokens = special_tokens + tokens

        return Vocab(tokens)

class TextDataset(Dataset):
    """文本数据集类"""
    def __init__(self, texts, vocab, max_seq_length):
        self.texts = texts
        self.vocab = vocab
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        print(text)
        tokens = text.split()[:self.max_seq_length - 2]  # 保留位置给起始/结束标记
        # 添加起始和结束标记
        tokens = ["<start>"] + tokens + ["<end>"]
        indices = self.vocab.lookup_indices(tokens)

        # 填充或截断
        if len(indices) < self.max_seq_length:
            indices += [self.vocab.pad_idx] * (self.max_seq_length - len(indices))
        else:
            indices = indices[:self.max_seq_length]

        return torch.tensor(indices, dtype=torch.long)