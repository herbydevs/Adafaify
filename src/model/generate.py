import torch

# simple greedy generation helper
@torch.no_grad()
def generate_greedy(model, tokenizer, encoder_text, max_len=64, device=None):
    """Encode input text, run encoder once, then step-decode greedily until EOS.

    model: TinyTransformer instance
    tokenizer: CharTokenizer (or compatible)
    encoder_text: raw string input
    max_len: max output tokens to produce (including SOS/EOS)
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    enc_ids = tokenizer.encode(encoder_text)
    enc = torch.tensor(enc_ids, dtype=torch.long, device=device).unsqueeze(0)  # (1, T_e)

    # Start decoder with SOS token
    dec_ids = [tokenizer.stoi["<SOS>"]]

    # Run encoder once
    # To keep shapes consistent, pad/truncate encoder to model.seq_len
    enc = enc[:, : model.seq_len]

    for _ in range(max_len - 1):
        dec_tensor = torch.tensor(dec_ids, dtype=torch.long, device=device).unsqueeze(0)
        # pad decoder to model.seq_len if desired, or allow variable length
        logits = model(enc, dec_tensor)  # (1, T_d, vocab)
        next_logits = logits[0, -1]  # last token logits
        next_id = int(torch.argmax(next_logits))
        if next_id == tokenizer.stoi.get("<EOS>"):
            break
        dec_ids.append(next_id)

    return tokenizer.decode(dec_ids)