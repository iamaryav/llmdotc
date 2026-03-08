import torch
import torch.nn as nn
import torch.nn.functional as F


# vanilla architecture is done
# work on improving the architecture

class GPTConfig:
    vocab_size: int = 50304
    hidden_size: int = 768
    max_seq_len: int = 2048
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    dropout: float = 0.2# change later
    bias: bool = False

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.hidden_size % config.num_attention_heads == 0
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_attention_heads
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, config.bias)
        self.k_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, config.bias)
        self.v_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, config.bias)
        self.out_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.register_buffer('tril', torch.tril(torch.ones(config.max_seq_len, config.max_seq_len)))


    def forward(self, x):
        # x -> (batch, max_seq_len, hidden_size)
        # do resize to implement the mha
        batch, seq_len = x.shape[:-1]
        # (batch, num_attention_heads, seq_len, head_dim)
        q = self.q_proj(x).view(batch, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)

        attn_wei = q @ k.transpose(-2, -1) * torch.sqrt(k.shape[-1]) ** 0.5
        attn_wei = attn_wei.masked_fill(self.tril[:seq_len,:seq_len] == 0, float("-inf"))
        attn_wei = F.softmax(attn_wei, dim=-1)  
        attn_wei = self.dropout(attn_wei) 

        attn_out = attn_wei @ v # (batch, num_attention_heads, seq_len, head_dim)
        attn_out = attn_out.transpose(1, 2).contiguous().reshape(batch, seq_len, -1)
        attn_out = self.out_proj(attn_out)
        return attn_out

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, config.bias)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(config.dropout)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, config.bias)

    def forward(self, x):
        y = self.dropout(self.down_proj(self.relu(self.up_proj(x))))
        return y

class DecoderLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(config.hidden_size)
        self.self_attn = CausalSelfAttention(config)
        self.post_attention_norm = nn.LayerNorm(config.hidden_size)
        self.ffn = MLP(config)

    def forward(self, x):
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.ffn(self.post_attention_norm(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        # change this to rot_emb
        self.pos_embed = nn.Embedding(config.max_seq_len, config.hidden_size)
        self.layers = nn.Sequential([DecoderLayer(config) for layer_idx in config.num_hidden_layers])
        self.norm = nn.LayerNorm(config.hidden_size) # change it to RMSNorm
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, config.bias)
        self._init_weights()

    def _init_weights(self):
        pass

    def forward(self, input_ids, target=None, device='cpu'):

        batch, seq_len = input_ids.shape
        tok_emb = self.embed_tokens(input_ids)
        pos_emb = self.pos_embed(torch.arange(seq_len, device=device))
        x = tok_emb + pos_emb
        x = self.layers(x)
        x = self.norm(x)
        logits = self.lm_head(x)

        if target == None:
            loss = None
        else:
            batch, seq_len, hidden_size = x.shape
            logits = logits.view(batch * seq_len, hidden_size)
            target = target.view(batch * seq_len)
            loss = F.cross_entropy(logits, target)

        return logits, loss













