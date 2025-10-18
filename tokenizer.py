import numpy as np
import re
from collections import Counter
import json
import os
import tiktoken


class BPETokenizer:
  """
  Defines a Byte Pair Encoding Tokenizer.
  Attributes:
    vocab_limit: Number of additional tokens (beyond 256) in the vocabulary.
    encoding_table: Defines mapping from a pair of merged tokens to the new token ID.
    decoding_tables: Defines mapping from a new token ID to the pair of merged tokens.
  """
  def __init__(self, 
               vocab_limit: int, 
               encoding_table: dict=None, 
               decoding_table: dict=None):
    """
    Creates an instance of the BPETokenizer class.
    Arguments:
      vocab_limit: Number of additional tokens (beyond 256) in the vocabulary.
      encoding_table: Defines mapping from a pair of merged tokens to the new token ID.
      decoding_tables: Defines mapping from a new token ID to the pair of merged tokens.
    """
    self.vocab_limit = vocab_limit
    self.encoding_table = encoding_table if encoding_table is not None else {}
    self.decoding_table = decoding_table if decoding_table is not None else {}

  def save(self, 
           tokenizer_dir: str="./tokenizer", 
           name: str="base")->None:
    """
    Saves the current vocab_limit and encoding_table to specified location.
    Arguments:
      tokenizer_dir: Directory where to save file.
      name: Name of json file to be save
    """
    os.makedirs(tokenizer_dir, exist_ok=True)
    name_enc = name + "encoding.json"
    enc_path = tokenizer_dir + "/" + name_enc
    model_dict = {}
    model_dict["vocab_limit"] = self.vocab_limit
    model_dict["merges"] = {f"{key[0]},{key[1]}":value for key, value in self.encoding_table.items()}
    with open(enc_path, 'w') as f:
      json.dump(model_dict, f)


  @classmethod
  def load(cls, 
           tokenizer_dir: str="./tokenizer", 
           name: str="base"):
    """
    Creates a new instance of the BPETokenizer class from saved data.
    Arguments:
      tokenizer_dir: Directory where to save file.
      name: Name of json file to be save
    """
    name_enc = name + "encoding.json"
    enc_path = tokenizer_dir + "/" + name_enc
    with open(enc_path, 'r') as f:
      model_dict = json.load(f)
    vocab_limit = model_dict["vocab_limit"]
    encoding_table = {tuple(int(s) for s in key.split(",")): v for key, v in model_dict["merges"].items()}
    decoding_table = {v:k for k,v in encoding_table.items()}
    return cls(vocab_limit, encoding_table, decoding_table)

  def train(self,
            text: str, 
            initial_vocab_size: int=256)->None:
    """
    Trains the instance of the BPETokenizer on given text.
    Arguments:
      text: text string to train the tokenizer,
      initial_vocab_size: Size of initial vocabulary, assumed to be 256.
    """
    current_encoding = text.encode("utf-8")
    current_encoding = list(map(int, current_encoding))
    for i in range(initial_vocab_size, self.vocab_limit):
      mcp = self._most_common_pair(current_encoding)
      if mcp is None:
        break
      current_encoding = self._replace(current_encoding, mcp, i)
      self.decoding_table[i], self.encoding_table[mcp] = mcp, i

  def encode(self, 
             text: str)->list:
    """
    Takes a string and tokenizes it.
    Arguments:
      text: string to be encoded.
    Returns:
      list of encoded tokens
    """
    encoded = text.encode("utf-8")
    encoded = list(map(int, encoded))
    while True:
      all_pairs = set(zip(encoded, encoded[1:]))
      for pair in self.encoding_table:
        if pair in all_pairs:
          encoded = self._replace(encoded, pair, self.encoding_table[pair])
          break
      else:
        break 
    return encoded

  def decode(self, 
             tokens: list)->str:
    """
    Takes a list of (valid) tokens and returns the decoded string.
    Arguments:
      tokens: list of valid tokens.
    Returns:
      string of decoded text.
    """
    decoded = []
    for token in tokens:
      self._unmerge(token, decoded)
    decoded_string = bytes(decoded).decode("utf-8", errors="replace")
    return decoded_string

  @staticmethod
  def _most_common_pair(l: list) -> tuple | None:
    if len(l) < 2:
      return None
    pair_counter = Counter(zip(l, l[1:]))
    return pair_counter.most_common(1)[0][0]

  @staticmethod
  def _replace(tokens: list, 
              pair_to_replace: tuple, 
              new_token: int) -> list:
    new_list = []
    if pair_to_replace is None:
      return tokens
    i = 0
    while i < len(tokens):
      if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == pair_to_replace:
        new_list.append(new_token)
        i += 2
      else:
        new_list.append(tokens[i])
        i += 1
    return new_list
  
  def _unmerge(self, token, results):
    #TODO: Use a stack to overcome recursion limits.
    if token < 256:
      results.append(token)
      return  
    else:
      t1, t2 = self.decoding_table[token]
      self._unmerge(t1, results)
      self._unmerge(t2, results)
      return 


class WordTokenizer:
  def __init__(self, text_2_token: dict=None):
    self.text_2_token = {} if text_2_token is None else text_2_token
    self.token_2_text = {token:text for text,token in text_2_token.items()} if text_2_token is not None else {}

  def train(self, text: str)->None:
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    preprocessed = [word.strip() for word in preprocessed if word.strip()]
    vocab_words = sorted(set(preprocessed))
    vocab_words.extend(["<|unk|>", "<|endoftext|>"])
    self.text_2_token = {word:idx for idx, word in enumerate(vocab_words)}
    self.token_2_text = {idx:word for idx, word in enumerate(vocab_words)}

  def encode(self, text: str)->list:
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    preprocessed = [word.strip() for word in preprocessed if word.strip()]
    return [self.text_2_token[word] if word in self.text_2_token else self.text_2_token["<|unk|>"] for word in preprocessed]

  def decode(self, tokens: list)->str:
    words = [self.token_2_text[idx] for idx in tokens]
    return re.sub(r'\s+([,.?!"()\'])', r'\1', " ".join(words))


class BPETokenizer_V1:
  """
  Basic Byte Pair Encoding Tokenizer.
  Attributes:
    decoder: hash map from token id to token pair ids.
    current_vocab_size: length of current vocab.
    vocab_limit: max lenn of vocab.
  """
  def __init__(self,
               current_vocab_len: int=256,
               vocab_limit: int=2048):
    self.decoder = {}
    self.current_vocab_size = current_vocab_len
    self.vocab_limit = vocab_limit

  def encode(self, text):
    """
    Encodes a string into a list of token IDs using the learned vocabulary.
    """
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
    """
    Decodes a list of token IDs back into a string.
    """
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
    """
    Trains the tokenizer on a given text, learning merge rules.
    Args:
      text (str): The text corpus to train on.
    """
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
