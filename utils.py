import torch 
import torch.nn as nn
import torch.nn.functional as F
import tiktoken


def generate_text_simple(model, 
                         idx, 
                         max_new_tokens, 
                         context_size):
  model.eval()
  for _ in range(max_new_tokens):
    input = idx[:, -context_size:]                        #[B, C]
    with torch.inference_mode():
      out = model(input)[:, -1:, :]                       #[B, C, V]->[B, 1, V]
    new_tokens = torch.argmax(out, dim=-1)                #[B, 1]
    idx = torch.cat((idx, new_tokens), dim=-1)
  return idx


def text_to_token_ids(text, 
                      tokenizer):
  return torch.tensor(tokenizer.encode(text, allowed_special={'<|endoftext|>'})).unsqueeze_(0)


def token_ids_to_text(ids, 
                      tokenizer):
  return tokenizer.decode(ids.squeeze_().tolist())


def calc_loss_batch(input_batch, 
                    target_batch, 
                    model, 
                    device):
  model.eval()
  with torch.inference_mode():
    out = model(input_batch.to(device)).flatten(0, 1)
  return F.cross_entropy(out, target_batch.flatten().to(device))


def calc_loss_loader(data_loader, 
                     model, 
                     device, 
                     num_batches=None):
  total_loss = 0.0
  if num_batches is None:
    num_batches = len(data_loader)
  else:
    num_batches = min(num_batches, len(data_loader))

  for i, (input_batch, target_batch) in enumerate(data_loader):
    if i < num_batches:
      total_loss += calc_loss_batch(input_batch, target_batch, model, device).item()
    else:
      break 
  return total_loss / num_batches


def train_model_simple(model, 
                       train_loader, 
                       val_loader, 
                       optimizer, 
                       device, 
                       num_epochs, 
                       start_context, 
                       tokenizer):
  train_losses, val_losses = [], []
  tokens_seen = 0

  for epoch in range(num_epochs):
    model.train()
    for batch_input, target_output in train_loader:
      batch_input, target_output = batch_input.to(device), target_output.to(device)
      out = model(batch_input)
      optimizer.zero_grad()
      loss = F.cross_entropy(out.flatten(0, 1), target_output.flatten())
      loss.backward()
      optimizer.step()
    train_loss = calc_loss_loader(train_loader, model, device)
    val_loss = calc_loss_loader(val_loader, model, device)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    print(f"Ep {epoch+1}: "
    f"Train loss {train_loss:.3f}, "
    f"Val loss {val_loss:.3f}")

    generate_and_print_sample(model, tokenizer, device, start_context)
  return train_losses, val_losses


def generate_and_print_sample(model, 
                                tokenizer, 
                                device, 
                                start_context: str="Once upon a time"):
  idx = text_to_token_ids(start_context, tokenizer)
  context_size = model.pos_emb.weight.shape[0]
  out = generate_text_simple(model, idx, 50, context_size)
  decoded_text = token_ids_to_text(out, tokenizer)
  print(decoded_text.replace("\n", " "))    

