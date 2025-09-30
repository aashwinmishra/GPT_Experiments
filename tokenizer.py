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


class BPETokenizer:
  def __init__(self,
               current_vocab_len: int=256,
               vocab_limit: int=2048):
    self.decoder = {}
    self.current_vocab_size = current_vocab_len
    self.vocab_limit = vocab_limit

  def encode(self, text):
    tokens = text.encode("utf-8")
    tokens = list(map(int, tokens))
    for idx in self.decoder:
      new_tokens = []
      i = 0
      while i < len(tokens):
        if i == len(tokens) - 1:
          new_tokens.append(tokens[i])
          break
        elif (tokens[i], tokens[i + 1]) == self.decoder[idx]:
          new_tokens.append(idx)
          i += 2
        else:
          new_tokens.append(tokens[i])
          i += 1
      tokens = new_tokens
    return tokens

  def decode(self, encoded: list):
    ans = encoded.copy()
    d = sorted(list(self.decoder.keys()), reverse=True)
    for token in d:
      new_ans = []
      for i in range(len(ans)):
        if ans[i] != token:
          new_ans.append(ans[i])
        else:
          new_ans.extend(self.decoder[token])
      ans = new_ans
    return ans

  def _train(self, text):
    tokens = text.encode("utf-8")
    tokens = list(map(int, tokens)) 
    tokens = self.merge(tokens, self.current_vocab_size, self.vocab_limit)
    return tokens

  @staticmethod
  def most_common_pair(tokens):
    counts = {}
    for key in zip(tokens[:-1], tokens[1:]):
      counts[key] = counts.get(key, 0) + 1
    return max(counts, key=counts.get)

  @staticmethod
  def replace(tokens, mcpair, new_token):
    new_tokens = []
    idx = 0
    while idx < len(tokens):
      if idx == len(tokens) - 1:
        new_tokens.append(tokens[idx])
        break
      if (tokens[idx], tokens[idx + 1]) == mcpair:
        new_tokens.append(new_token)
        idx += 2
      else:
        new_tokens.append(tokens[idx])
        idx += 1
    return new_tokens

  def merge(self, 
            tokens, 
            current_vocab_len, 
            vocab_len_limit):
    for idx in range(current_vocab_len, vocab_len_limit):
      mcpair = self.most_common_pair(tokens)
      tokens = self.replace(tokens, mcpair, idx)
      self.decoder[idx] = mcpair
    return tokens
