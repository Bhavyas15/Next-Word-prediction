import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
  def __init__(self, d_model, max_len, dropout_prob=0.1):
    super().__init__()
    self.dropout = nn.Dropout(p = dropout_prob)

    position = torch.arange(max_len).unsqueeze(1)
    i = torch.arange(0, d_model//2)
    pe = torch.zeros(1, max_len, d_model)
    pe[:, :, 0::2] = torch.sin(position / (10000)**(2*i/d_model))
    pe[:, :, 1::2] = torch.cos(position / (10000)**(2*i/d_model))
    self.register_buffer('pe', pe)
  def forward(self, x):
    # x.shape : N x T x D 
    x = x + self.pe[:, :x.size(1), :]
    return self.dropout(x)

class TransformerBlock(nn.Module):
  def __init__(self, d_k, d_model, n_heads, max_len, dropout_prob = 0.1):
    super().__init__()

    self.n1 = nn.LayerNorm(d_model,eps=1e-6)
    self.n2 = nn.LayerNorm(d_model,eps=1e-6)

    self.mha=torch.nn.MultiheadAttention(d_model, n_heads, dropout=dropout_prob, batch_first=True)

    self.ann = nn.Sequential(
        nn.Linear(d_model, d_model*4),
        nn.GELU(),
        nn.Linear(d_model*4, d_model),
        nn.Dropout(dropout_prob),
    )

    self.dropout = nn.Dropout(p=dropout_prob)

  def forward (self, x, mask):
    batch_size, seq_len, _ = x.size()
    attn_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).to(x.device)
    attn_mask = attn_mask.masked_fill(attn_mask == 1, float('-inf'))  # Convert to -inf for softmax

    attn_output, attn_output_weights = self.mha(x,x,x,attn_mask=attn_mask, key_padding_mask=mask)
    x = self.n1(x + attn_output)
    x = self.n2(x + self.ann(x))
    x = self.dropout(x)
    return x

class Decoder(nn.Module):
  def __init__(self, vocab_size, max_len, d_k, d_model, n_heads, n_layers, dropout_prob):
    super().__init__()

    self.embedding = nn.Embedding(vocab_size, d_model)
    self.pos_encoding=PositionalEncoding(d_model, max_len, dropout_prob)

    self.n_heads=n_heads
    transformer_blocks=[TransformerBlock(d_k, d_model, n_heads, max_len, dropout_prob) for _ in range(n_layers) ]

    self.transformer_blocks = nn.Sequential(*transformer_blocks)

    self.ln = nn.LayerNorm(d_model)
    self.fc = nn.Linear(d_model, vocab_size)

  def forward(self, x, mask):
    x=self.embedding(x)
    x=self.pos_encoding(x)

    for block in self.transformer_blocks:
      block.to(x.device)
      x = block(x, mask)
    x = self.ln(x)
    x = self.fc(x)

    return x