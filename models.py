import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
  def __init__(self, 
               embed_dim: int, 
               context_length: int, 
               num_heads: int, 
               dropout: float, 
               qkv_bias: bool=False):
    super().__init__()
    assert embed_dim % num_heads == 0
    self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=qkv_bias)
    self.Wo = nn.Linear(embed_dim, embed_dim)
    self.dropout = nn.Dropout(dropout)
    self.mask = nn.Parameter(torch.triu(torch.ones(context_length, context_length), diagonal=1))
    self.num_heads = num_heads
    self.head_dim = int(embed_dim / num_heads)

  def forward(self, x: torch.tensor)->torch.tensor:
    B, S, D = x.shape
    q, k, v = self.Wqkv(x).chunk(3, dim=-1)                       #[B, S, D]
    q = q.reshape((B, S, self.num_heads, self.head_dim))          #[B, S, n, d]
    k = k.reshape((B, S, self.num_heads, self.head_dim))          #[B, S, n, d]
    v = v.reshape((B, S, self.num_heads, self.head_dim))          #[B, S, n, d]
    q = q.permute((0, 2, 1, 3))                                   #[B, n, S, d]
    k = k.permute((0, 2, 3, 1))                                   #[B, n, d, S]
    v = v.permute((0, 2, 1, 3))                                   #[B, n, S, d]
    attention_scores = q @ k / (self.head_dim)**0.5               #[B, n, S, S]
    attention_scores.masked_fill_(self.mask[:S, :S ].bool(), -torch.inf)
    attention_weights = torch.softmax(attention_scores, dim=-1)
    attention_weights = self.dropout(attention_weights  )
    attention = attention_weights @ v                             #[B, n, S, d]
    attention = attention.permute((0, 2, 1, 3)).reshape(B, S, D)  #[B, S, D]
    return self.Wo(attention) 


class LayerNorm(nn.Module):
  def __init__(self, 
               embed_dim: int=768, 
               eps: float=1e-5):
    super().__init__()
    self.scale = nn.Parameter(torch.ones((embed_dim,)))
    self.shift = nn.Parameter(torch.zeros((embed_dim,)))
    self.eps = eps 
  
  def forward(self, x: torch.tensor)->torch.tensor:
    return self.shift + self.scale * (x - torch.mean(x, dim=-1, keepdim=True)) / (torch.std(x, dim=-1, keepdim=True) + self.eps)


class GeLU(nn.Module):
  def __init__(self):
    super().__init__()
    self.add_term = torch.sqrt(torch.tensor(2/torch.pi))

  def forward(self, x: torch.tensor)->torch.tensor:
    return 0.5 * x * (1.0 + torch.tanh(self.add_term * (x + 0.044715 * torch.pow(x, 3)))) 


class MLP(nn.Module):
  def __init__(self, embed_dim: int=768):
    super().__init__()
    self.mlp = nn.Sequential(nn.Linear(embed_dim, 4 * embed_dim),
                             GeLU(),
                             nn.Linear(4 * embed_dim, embed_dim))
    
  def forward(self, x: torch.tensor)->torch.tensor:
    return self.mlp(x)


class TransformerBlock(nn.Module):
  def __init__(self, 
               embed_dim: int=768, 
               context_length: int=512, 
               num_heads: int=12, 
               dropout: float=0.1, 
               qkv_bias: bool=False):
    super().__init__()
    self.norm1 = LayerNorm(embed_dim)
    self.mhsa = MultiHeadSelfAttention(embed_dim, context_length, num_heads, dropout, qkv_bias)
    self.dropout1 = nn.Dropout(dropout)
    self.norm2 = LayerNorm(embed_dim)
    self.mlp = MLP(embed_dim)
    self.dropout2 = nn.Dropout(dropout)

  def forward(self, x: torch.tensor)->torch.tensor:
    x = x + self.dropout1(self.mhsa(self.norm1(x)))
    return x + self.dropout2(self.mlp(self.norm2(x)))

