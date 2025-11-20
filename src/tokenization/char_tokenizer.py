class CharTokenizer:
    def __init__(self):
        # Fixed special tokens
        self.special_tokens = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]

        self.stoi = {}
        self.itos = {}
        self._vocab_built = False

    def fit(self, text: str) -> None:
        """Build vocabulary from characters in text."""
        # Extract all unique chars from input
        chars = sorted(list(set(text)))

        # Build complete vocab with special tokens first
        vocab = self.special_tokens + chars

        # Build stoi/itos mappings
        self.stoi = {ch: i for i, ch in enumerate(vocab)}
        self.itos = {i: ch for i, ch in enumerate(vocab)}

        self._vocab_built = True

    def encode(self, text: str) -> list[int]:
        """Encode text into token IDs, adding SOS/EOS and using UNK for unknown chars."""
        if not self._vocab_built:
            raise ValueError("Tokenizer vocabulary not built. Call fit(text).")

        tokens = []

        # Add start token
        tokens.append(self.stoi["<SOS>"])

        # Encode characters
        for ch in text:
            if ch in self.stoi:
                tokens.append(self.stoi[ch])
            else:
                tokens.append(self.stoi["<UNK>"])

        # Add end token
        tokens.append(self.stoi["<EOS>"])

        return tokens

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs back to text, stopping at EOS."""
        if not self._vocab_built:
            raise ValueError("Tokenizer vocabulary not built. Call fit(text).")

        result = ""

        for i in ids:
            token = self.itos.get(i, "<UNK>")

            # Stop if EOS is reached
            if token == "<EOS>":
                break

            # Skip padding and SOS markers
            if token in ("<PAD>", "<SOS>"):
                continue

            result += token

        return result

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)
