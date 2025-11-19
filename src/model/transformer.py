import torch
import torch.nn as nn
import torch.nn.functional as F


class Transformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim = 64, seq_len = 32, ff_hidden_dim = 128):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
