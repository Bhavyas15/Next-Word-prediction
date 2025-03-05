import torch 
import torch.nn as nn
from transformers import AutoTokenizer
from datasets import load_dataset

from model_architecture import Decoder
from functions import initialize_model, download_model
from UI import ui_ux

checkpoint = 'distilbert-base-cased'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

model_init = Decoder (
    vocab_size = tokenizer.vocab_size,
    max_len=tokenizer.model_max_length,
    d_k=32,
    d_model = 512,
    n_heads = 8, 
    n_layers = 12, 
    dropout_prob =0.1
)
optimizer_init = torch.optim.AdamW(model_init.parameters(),lr=1e-4) 
criterion_init = nn.CrossEntropyLoss(ignore_index = tokenizer.pad_token_id)

path='data/pred_model.pth'
if not path:
    download_model()
model,optimizer=initialize_model(model_init, optimizer_init, path, load_dataset, tokenizer, criterion_init)

ui_ux(model,tokenizer)

