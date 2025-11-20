# scripts/train.py
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from tqdm import tqdm

# scripts/train.py
from model.transformer import TinyTransformer
from tokenization import CharTokenizer  # Make sure tokenizer.py exists here


# -------------------------
# Config
SEQ_LEN = 64
BATCH_SIZE = 2
EMBED_DIM = 64
FF_HIDDEN = 128
LR = 1e-3
EPOCHS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# Load dataset
df = pd.read_csv("./data/raw/train.csv")
inputs = df["input"].tolist()
summaries = df["summary"].tolist()

# -------------------------
# Build tokenizer on both inputs and summaries
tokenizer = CharTokenizer()
all_text = " ".join(inputs + summaries)
tokenizer.fit(all_text)

# Add special tokens
for token in ["<SOS>", "<EOS>"]:
    if token not in tokenizer.stoi:
        idx = len(tokenizer.stoi)
        tokenizer.stoi[token] = idx
        tokenizer.itos[idx] = token

VOCAB_SIZE = tokenizer.vocab_size

# -------------------------
# Build model
model = TinyTransformer(vocab_size=VOCAB_SIZE, embedding_dim=EMBED_DIM, seq_len=SEQ_LEN).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# -------------------------
# Helper: prepare batch
def encode_pair(input_text, target_text):
    enc_ids = tokenizer.encode(input_text)[:SEQ_LEN]
    dec_ids = [tokenizer.stoi["<SOS>"]] + tokenizer.encode(target_text)[:SEQ_LEN-2] + [tokenizer.stoi["<EOS>"]]
    
    # pad sequences
    enc_ids += [0]*(SEQ_LEN - len(enc_ids))
    dec_ids += [0]*(SEQ_LEN - len(dec_ids))
    return torch.tensor(enc_ids, dtype=torch.long), torch.tensor(dec_ids, dtype=torch.long)

# -------------------------
# Training loop
for epoch in range(EPOCHS):
    epoch_loss = 0
    for i in tqdm(range(0, len(inputs), BATCH_SIZE)):
        batch_inputs = inputs[i:i+BATCH_SIZE]
        batch_summaries = summaries[i:i+BATCH_SIZE]

        enc_batch = []
        dec_batch = []
        for inp, out in zip(batch_inputs, batch_summaries):
            enc_ids, dec_ids = encode_pair(inp, out)
            enc_batch.append(enc_ids)
            dec_batch.append(dec_ids)

        enc_batch = torch.stack(enc_batch).to(DEVICE)
        dec_batch = torch.stack(dec_batch).to(DEVICE)

        optimizer.zero_grad()
        logits = model(enc_batch, dec_batch[:, :-1])
        loss = criterion(logits.reshape(-1, VOCAB_SIZE), dec_batch[:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    

    print(f"Epoch {epoch+1}/{EPOCHS} Loss: {epoch_loss/len(inputs):.4f}")

# -------------------------
# Save model
torch.save({
    "model_state_dict": model.state_dict(),
    "tokenizer": tokenizer.stoi
}, "model_checkpoint.pt")

print("Training complete, checkpoint saved as model_checkpoint.pt")
