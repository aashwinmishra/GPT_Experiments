import torch
import tiktoken


def text_to_token_ids(text, tokenizer):
  return torch.tensor(tokenizer.encode(text)).unsqueeze_(0)

def token_ids_to_text(token_ids, tokenizer):
  return tokenizer.decode(token_ids.squeeze_().tolist())

def generate_sampled_text(model,
                          idx: torch.tensor,
                          max_new_tokens: int=25,
                          context_length: int=256,
                          temperature: float = 1.0,
                          k: int = 10,
                          device: torch.device=torch.device("cpu")):
  model.eval()
  idx = idx.to(device)
  for _ in range(max_new_tokens):
    with torch.inference_mode():
      logits = model(idx[:, -min(idx.shape[-1], context_length):])[0, -1, :] / temperature
      top_k_values, _ = torch.topk(logits, k)
      cutoff = top_k_values[-1]
      logits = torch.where(logits < cutoff, -float("inf"), logits)
    dist = torch.distributions.Categorical(logits=logits)
    new_idx = dist.sample().view(1, 1)
    idx = torch.cat([idx, new_idx], dim=-1)
  return idx


def generate_text(input_text,
                  tokenizer,
                  max_length,
                  context_length,
                  model,
                  device=torch.device('cpu')
                  ):
  ids = tokenizer.encode(input_text)                                            #[s] (list)
  ids = torch.tensor(ids).to(device).unsqueeze_(0)                              #[1,s]
  for i in range(max_length):
    with torch.inference_mode():
      out = model(ids[:, -context_length:])[:, -1, :].argmax(dim=-1, keepdim=True)#[1, s, 50257] -> [1, 1]
    ids = torch.cat([ids, out], dim=-1)
  return tokenizer.decode(ids.squeeze_().tolist())


def save_model_opt(model, opt, dir: str="./"):
  torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": opt.state_dict(),
    },
    dir + "model_and_optimizer.pth")


def load_model_opt(dir, device=torch.device("cpu")):
  return torch.load(dir + "model_and_optimizer.pth", map_location=device)

