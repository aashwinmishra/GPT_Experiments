import torch
import torch.nn as nn


class LayerNorm(nn.Module):
  def __init__(self,
               emb_dim: int=768,
               eps: float=1e-5):
    super().__init__()
    self.scale = nn.Parameter(torch.ones(emb_dim))
    self.shift = nn.Parameter(torch.zeros(emb_dim))
    self.eps = eps

  def forward(self, x):
    return self.shift + self.scale * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps)


class GELU(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, x):
    return 0.5 * x *(1 + torch.tanh((x + 0.044715 * torch.pow(x, 3)) * (2/torch.pi) ** 0.5))


class FeedForward(nn.Module):
  def __init__(self, emb_dim: int=768):
    super().__init__()
    self.layers = nn.Sequential(
        nn.Linear(emb_dim, 4 * emb_dim),
        GELU(),
        nn.Linear(4 * emb_dim, emb_dim)
    )

  def forward(self, x):
    return self.layers(x)


class MHSA(nn.Module):
  def __init__(self,
               d: int=768,
               context_length: int=1024,
               n_heads: int=12,
               dropout: float=0.1,
               qkv_bias: bool=False
               ):
    assert d % n_heads == 0
    super().__init__()
    self.W_qkv = nn.Linear(d, 3 * d, bias=qkv_bias)
    self.register_buffer("mask", torch.triu(torch.ones((context_length, context_length)), diagonal=1).bool())
    self.dropout = nn.Dropout(dropout)
    self.Wo = nn.Linear(d, d)
    self.n_heads = n_heads
    self.head_dim = d // n_heads

  def forward(self, x):
    b, s, d = x.shape
    Q, K, V = self.W_qkv(x).chunk(3, dim=-1)
    Q = torch.reshape(Q, (b, s, self.n_heads, self.head_dim)).transpose(1, 2)   #[b, nh, s, dh]
    K = torch.reshape(K, (b, s, self.n_heads, self.head_dim)).permute(0,2,3,1)  #[b, nh, dh, s]
    V = torch.reshape(V, (b, s, self.n_heads, self.head_dim)).transpose(1, 2)   #[b, nh, s, dh]
    attention_scores = Q @ K / self.head_dim ** 0.5
    attention_scores.masked_fill_(self.mask[:s, :s], -torch.inf)
    attention_weights = torch.softmax(attention_scores, dim=-1)
    attention_weights = self.dropout(attention_weights)
    attention = attention_weights @ V                                           #[b, nh, s, dh]
    attention = torch.reshape(attention.transpose(1, 2), (b, s, d))
    return self.Wo(attention)


class TransformerBlock(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    self.norm1 = LayerNorm(cfg["emb_dim"])
    self.attention = MHSA(d=cfg["emb_dim"],
                          context_length=cfg["context_length"],
                          n_heads=cfg["n_heads"],
                          dropout=cfg["drop_rate"],
                          qkv_bias=cfg["qkv_bias"])
    self.drop1 = nn.Dropout(cfg["drop_rate"])
    self.norm2 = LayerNorm(cfg["emb_dim"])
    self.feedforward = FeedForward(cfg["emb_dim"])
    self.drop2 = nn.Dropout(cfg["drop_rate"])

  def forward(self, x):
    x = x + self.drop1(self.attention(self.norm1(x)))
    return x + self.drop2(self.feedforward(self.norm2(x)))


class GPTModel(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
    self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
    self.drop_emb = nn.Dropout(cfg["drop_rate"])
    self.trf_blks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
    self.final_norm = LayerNorm(cfg["emb_dim"])
    self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

  def forward(self, x):
    tok_emb = self.tok_emb(x)
    pos_emb = self.pos_emb(torch.arange(x.shape[-1], device=x.device))
    x = tok_emb + pos_emb
    x = self.drop_emb(x)
    x = self.trf_blks(x)
    x = self.final_norm(x)
    return self.out_head(x)

