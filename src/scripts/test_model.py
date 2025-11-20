import torch
import csv
from model.transformer import TinyTransformer
from tokenization import CharTokenizer
from model.generate import generate_greedy
import os

# -------------------------
# Load checkpoint
checkpoint = torch.load("model_checkpoint.pt", map_location="cpu")

# -------------------------
# Recreate tokenizer
tokenizer = CharTokenizer()
tokenizer.stoi = checkpoint["tokenizer"]
tokenizer.itos = {i: ch for ch, i in tokenizer.stoi.items()}
tokenizer._vocab_built = True

# -------------------------
# Recreate model
vocab_size = len(tokenizer.stoi)
seq_len = 64
model = TinyTransformer(vocab_size=vocab_size, seq_len=seq_len)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# -------------------------
# Prepare results folder
os.makedirs("results", exist_ok=True)
results_file = os.path.join("results", "generated_results.txt")

# -------------------------
# Load test sentences
test_csv = os.path.join("data", "raw", "test_sentences.csv")
sentences = []
with open(test_csv, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        sentences.append(row["sentence"])

# -------------------------
# Generate summaries and write results
with open(results_file, "w") as f:
    f.write("Input sentence | Generated summary | Accuracy\n")
    for sentence in sentences:
        generated = generate_greedy(model, tokenizer, sentence, max_len=50)
        
        # Dummy "accuracy" for now: simple character match %
        length = max(len(sentence), len(generated))
        matches = sum(1 for a, b in zip(sentence, generated) if a == b)
        accuracy = matches / length if length > 0 else 0.0

        f.write(f"{sentence} | {generated} | {accuracy:.2f}\n")

print(f"Results written to {results_file}")
