import numpy as np
import re
import tiktoken


class WordLevelTokenizer:
  def __init__(self, raw_text: str) -> None:
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    preprocessed = [token for token in preprocessed if token.strip()]
    vocab = np.unique(preprocessed).tolist()
    vocab.extend(["<unk>", "<sos>", "<eos>"])
    self.id_2_token = {idx:token for idx, token in enumerate(vocab)}
    self.token_2_id = {token: idx for idx, token in enumerate(vocab)}

  def __len__(self) -> int:
    return len(self.token_2_id.keys())

  def encode(self, text: str) -> list:
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    preprocessed = [token for token in preprocessed if token.strip()]
    encoded = [self.token_2_id[token] if token in self.token_2_id else self.token_2_id["<unk>"] for token in preprocessed]
    return [self.token_2_id["<sos>"]] + encoded + [self.token_2_id["<eos>"]]

  def decode(self, ids: list) -> str:
    decoded = [self.id_2_token[id] for id in ids]
    text = " ".join(decoded)
    return re.sub(r'\s+([,.?!"()\'])', r'\1', text)

