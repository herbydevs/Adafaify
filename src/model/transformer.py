import torch
import torch.nn as nn
from .attention import CausalSelfAttention

class TinyTransformer(nn.Module):
    """
    Minimal encoder-decoder transformer for text summarization.
    - Shared token & positional embeddings
    - Encoder: standard self-attention
    - Decoder: causal self-attention + cross-attention
    """
    def __init__(
        self,
        vocab_size,
        embedding_dim=64,
        seq_len=64,
        encoder_layers=2,
        decoder_layers=2,
        heads=4,
        ff_hidden_dim=128,
        dropout=0.1
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.seq_len = seq_len

        # Token & positional embeddings
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embedding = nn.Embedding(seq_len, embedding_dim)

        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            nn.ModuleDict({
                "self_attn": CausalSelfAttention(embedding_dim, num_heads=heads, causal=False, dropout=dropout),
                "norm1": nn.LayerNorm(embedding_dim),
                "ff": nn.Sequential(
                    nn.Linear(embedding_dim, ff_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(ff_hidden_dim, embedding_dim)
                ),
                "norm2": nn.LayerNorm(embedding_dim),
            })
            for _ in range(encoder_layers)
        ])

        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            nn.ModuleDict({
                "self_attn": CausalSelfAttention(embedding_dim, num_heads=heads, causal=True, dropout=dropout),
                "norm1": nn.LayerNorm(embedding_dim),
                "cross_attn": CausalSelfAttention(embedding_dim, num_heads=heads, causal=False, dropout=dropout),
                "norm2": nn.LayerNorm(embedding_dim),
                "ff": nn.Sequential(
                    nn.Linear(embedding_dim, ff_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(ff_hidden_dim, embedding_dim)
                ),
                "norm3": nn.LayerNorm(embedding_dim),
            })
            for _ in range(decoder_layers)
        ])

        # Output projection
        self.output = nn.Linear(embedding_dim, vocab_size)

    def forward(self, encoder_input, decoder_input):
        """
        encoder_input: (B, T_e) token IDs
        decoder_input: (B, T_d) token IDs (shifted right during training)
        Returns logits: (B, T_d, vocab_size)
        """
        B, T_e = encoder_input.shape
        B2, T_d = decoder_input.shape
        assert B == B2, "Batch size mismatch"

        # Embeddings + positional encoding
        enc_emb = self.token_embedding(encoder_input) + self.pos_embedding(
            torch.arange(T_e, device=encoder_input.device).unsqueeze(0)
        )
        dec_emb = self.token_embedding(decoder_input) + self.pos_embedding(
            torch.arange(T_d, device=decoder_input.device).unsqueeze(0)
        )

        # --- Encoder ---
        h = enc_emb
        for layer in self.encoder_layers:
            attn = layer["self_attn"](h, h, h)
            h = layer["norm1"](h + attn)
            ff = layer["ff"](h)
            h = layer["norm2"](h + ff)
        memory = h  # (B, T_e, C)

        # --- Decoder ---
        out = dec_emb
        for layer in self.decoder_layers:
            # self-attention (causal)
            self_attn = layer["self_attn"](out, out, out)
            out = layer["norm1"](out + self_attn)

            # cross-attention
            cross_attn = layer["cross_attn"](memory, memory, out)
            out = layer["norm2"](out + cross_attn)

            # feed-forward
            ff = layer["ff"](out)
            out = layer["norm3"](out + ff)

        logits = self.output(out)  # (B, T_d, vocab_size)
        return logits
