class CharTokenizer:
    def __init__(self):
        self.stoi = {}
        self.itos = {}
        self._vocab_built = False 

    


    def fit(self, text: str) -> None:
        chars = sorted(list(set(chars)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos =  {i : ch for i, ch in enumerate(chars)}
        self._vocab_built = True

    
    def encode(self, text: str) -> list[int]:
        if not self._vocab_built:
            raise ValueError("Tokenizer vocabulary not built call fit(tex)")
        return [self.stoi[ch] for ch in text if ch in self.stoi]

    def decode(self, ids: list[int]) -> str:
        if not self._vocab_built:
            raise  ValueError("Tokenizer vocabulary not built call fit(tex)")
        return "".join(self.itos[i] for i in ids)
    

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)
        