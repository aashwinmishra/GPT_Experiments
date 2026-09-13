import re
import tiktoken
from datasets import load_dataset
import torch
import torch.nn as nn


class WordTokenizer:
  def __init__(self, raw_text: str):
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    vocab = list(set(preprocessed))
    vocab.sort()
    self.id2token = {id: token for id,token in enumerate(vocab)}
    self.id2token[len(vocab)] = "<unk>"
    self.token2id = {token:id for id, token in self.id2token.items()}

  def encode(self, text):
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    return [self.token2id[token] if token in self.token2id else self.token2id["<unk>"] for token in preprocessed]

  def decode(self, ids):
    text = [self.id2token[id] for id in ids]
    text = " ".join(text)
    return re.sub(r'\s+([,.?!"()\'])', r'\1', text)


class GPTDataset(torch.utils.data.Dataset):
  def __init__(self, 
               tokenized_data, 
               context_length, 
               stride):
    self.data = torch.tensor(tokenized_data, dtype=torch.long)
    self.context_length = context_length
    self.stride = stride
    self.num_chunks = (len(self.data) - context_length) // stride

  def __len__(self):
    return self.num_chunks

  def __getitem__(self, idx):
    start_idx = idx * self.stride
    return self.data[start_idx : start_idx + self.context_length], self.data[start_idx + 1 : start_idx + self.context_length + 1]


def encode_text(text_list, tokenizer):
  text = "\n".join(text_list)
  return tokenizer.encode_ordinary(text)


def create_dataloaders(batch_size=4,
                      context_length=64,
                      stride=32,
                      drop_last=True,
                      num_workers=2):
  tokenizer = tiktoken.get_encoding("gpt2")
  dataset = load_dataset('Salesforce/wikitext', "wikitext-2-raw-v1")
  train_tokens = encode_text(dataset["train"]["text"], tokenizer)
  val_tokens = encode_text(dataset["validation"]["text"], tokenizer)
  train_ds = GPTDataset(train_tokens, context_length, stride)
  val_ds = GPTDataset(val_tokens, context_length, stride)
  train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=drop_last, num_workers=num_workers, pin_memory=True)
  val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=drop_last, num_workers=num_workers, pin_memory=True)
  return train_loader, val_loader

