import torch
import numpy as np
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader
from datetime import datetime

device =torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def tokenize_fn(batch,tokenizer):
  return tokenizer(batch['target'], truncation = True)

def train_load(common_allen, tokenizer):
    tokenized_datasets = common_allen.map(lambda batch: tokenize_fn(batch, tokenizer), batched = True)
    data_collator = DataCollatorWithPadding(tokenizer = tokenizer)

    tokenized_datasets = tokenized_datasets.remove_columns(["concept_set_idx", "concepts", "target"])

    train_loader =  DataLoader(tokenized_datasets["train"], shuffle = True, batch_size = 32, collate_fn = data_collator)
    return train_loader

def train(model, criterion, optimizer, train_loader, epochs):
  train_losses = np.zeros(epochs)
  for it in range(epochs):
    model.train()
    t0 = datetime.now()
    train_loss = []
    for batch in train_loader:
      batch = {k:v.to(device) for k, v in batch.items()}

      # zero the parameter gradients
      optimizer.zero_grad()

      # shift the targets backwards
      targets = batch['input_ids'].clone().detach()
      targets = targets[:, 1:].contiguous() 

      # forward pass
      batch['attention_mask'] = batch['attention_mask'].float().masked_fill(batch['attention_mask'] == 0, float('-inf'))
      batch['attention_mask'] = batch['attention_mask'].masked_fill(batch['attention_mask'] == 1, 0)

      outputs = model(batch['input_ids'], batch['attention_mask'].to(bool))
      logits = outputs[:, :-1, :]  # Align with shifted targets   # changed
      loss = criterion(logits.transpose(2, 1), targets)

      # Backward and optimize
      loss.backward()

       # Gradient Clipping: Add this line to clip gradients
      torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

      optimizer.step()
      train_loss.append(loss.item())

    # train Loss and test loss
    train_loss = np.mean(train_loss)

    # save lossess
    train_losses[it] = train_loss

    dt = datetime.now() - t0
    print(f'Epoch: {it+1}/{epochs}, Train Loss: {train_loss:.4f}, Duration: {dt}')

  return train_losses

def save_model(model,optimizer):
    save_path = "data/pred_model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, save_path)
    print("Model saved successfully!")
