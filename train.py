import torch
import torch.nn as nn
from utils import generate_text
import tiktoken


def train_step(model, train_dl, loss_fn, opt, device):
  model.train()
  losses = 0.0
  count = 0
  for inputs, targets in train_dl:
    opt.zero_grad()
    inputs, targets = inputs.to(device), targets.to(device)
    logits = model(inputs)
    loss = loss_fn(logits.flatten(0,1), targets.flatten())
    loss.backward()
    opt.step()
    losses += loss.cpu().item()
    count += 1
  return losses / count


def val_step(model, val_dl, loss_fn, device):
  model.eval()
  losses = 0.0
  count = 0
  for inputs, targets in val_dl:
    inputs, targets = inputs.to(device), targets.to(device)
    with torch.inference_mode():
      logits = model(inputs)
      loss = loss_fn(logits.flatten(0,1), targets.flatten())
      losses += loss.cpu().item()
      count += 1
  return losses / count

def train(model, train_dl, val_dl, loss_fn, opt, num_epochs, device):
  train_losses, val_losses = [], []
  for epoch in range(num_epochs):
    train_loss = train_step(model, train_dl, loss_fn, opt, device)
    val_loss = val_step(model, val_dl, loss_fn, device)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    print(f"Epoch: {epoch+1} Train Loss: {train_loss:.5f} Val Loss: {val_loss:.5f}")
    text = generate_text("Once upon a time", tiktoken.get_encoding("gpt2"), 25, 256, model, device)
    print(text)
  return train_losses, val_losses

