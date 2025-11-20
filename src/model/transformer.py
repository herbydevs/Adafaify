import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, seq_len=32, ff_hidden_dim=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.seq_len = seq_len

        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)

        # Positional embeddings
        self.pos_embedding = nn.Embedding(seq_len, embedding_dim)

        # Single attention head
        self.attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=1, batch_first=True)

        # Feed-forward layer
        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, embedding_dim)
        )

        # Output projection to vocab size
        self.output = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        # x shape: (batch_size, seq_len)
        batch_size, seq_len = x.shape

        # Token embeddings
        token_emb = self.token_embedding(x)  # (batch_size, seq_len, embedding_dim)
