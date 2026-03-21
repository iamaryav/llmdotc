import torch
import torch.nn as nn
import torch.nn.functional as F


# vanilla architecture is done
# work on improving the architecture

class GPTConfig:
    vocab_size: int = 50304
    hidden_size: int = 768
    intermediate_size: int = 768 * 4
    max_seq_len: int = 2048
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    num_kv_heads: int = 2
    dropout: float = 0.2
    bias: bool = False

def norm(x):
    return F.rms_norm(x, (x.size(-1),))

def apply_rotary_emb(x, cos, sin):
    # x1 * cos - x2 * sin, x1 sin + x2 cos
    # embedding -> Q/K Proj -> RoPE rotation -> attention
    # (batch, num_attention_heads, seq_len, head_dim)
    assert x.ndim == 4
    d = x.shape[3] // 2

    # half split
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos

    return torch.cat([y1, y2], dim=-1)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.hidden_size % config.num_attention_heads == 0
        assert config.num_attention_heads % config.num_kv_heads == 0
        self.num_attention_heads = config.num_attention_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.hidden_size // self.num_attention_heads
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, config.bias)
        self.k_proj = nn.Linear(config.hidden_size, config.num_kv_heads * self.head_dim, config.bias)
        self.v_proj = nn.Linear(config.hidden_size, config.num_kv_heads * self.head_dim, config.bias)
        self.out_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.register_buffer('tril', torch.tril(torch.ones(config.max_seq_len, config.max_seq_len)))


    def forward(self, x, cos, sin):
        # x -> (batch, max_seq_len, hidden_size)
        batch, seq_len = x.shape[:-1]
        # (batch, num_attention_heads, seq_len, head_dim)
        q = self.q_proj(x).view(batch, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        # expand the kv_heads to match query dimensions for calculation
        k = k.expand(batch, self.num_attention_heads, seq_len, self.head_dim)
        v = v.expand(batch, self.num_attention_heads, seq_len, self.head_dim)

        # apply rotary embedding
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # QKNorm with rmsnorm
        # rms norm to both query and key
        q = norm(q) 
        k = norm(k)

        attn_wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
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
        # Gate Linear unit with relu in between
        # Will use other activation in between
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, config.bias)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, config.bias)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(config.dropout)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, config.bias)

    def forward(self, x):
        # (act(xw) * xv)w2
        y = self.dropout(self.down_proj(self.relu(self.gate_proj(x)) * self.up_proj(x)))
        return y

class DecoderLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(config.hidden_size)
        self.self_attn = CausalSelfAttention(config)
        self.post_attention_norm = nn.LayerNorm(config.hidden_size)
        self.ffn = MLP(config)

    def forward(self, x, cos, sin):
        # pre-attention layernorm -> self attention -> post-attention layernorm -> ffn
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        x = x + self.ffn(self.post_attention_norm(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([DecoderLayer(config) for layer_idx in range(config.num_hidden_layers)])
        self.norm = nn.LayerNorm(config.hidden_size) # change it to RMSNorm
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, config.bias)
        self._init_weights()

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        # stride the channels
        if device is None:
            device = self.embed_tokens.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))

        # stride the time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # calculate the rotation frequencies at each (time, channel) pair
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()

        # add batch and head dimension for broadcasting
        # shape: (1, 1, seq_len, head_dim//2)
        cos, sin = cos[None, None, :, :], sin[None, None, :, :]
        return cos, sin

    def _init_weights(self):
        pass

    def forward(self, input_ids, target=None, device='cpu'):

        batch, seq_len = input_ids.shape
        tok_emb = self.embed_tokens(input_ids)

        # precompute rotary embeddings
        head_dim = self.config.hidden_size // self.config.num_attention_heads
        cos, sin = self._precompute_rotary_embeddings(seq_len, head_dim, device=device)

        x = tok_emb
        for layer in self.layers:
            x = layer(x, cos, sin)
        x = self.norm(x)
        logits = self.lm_head(x)

        if target == None:
            loss = None
        else:
            batch, seq_len, vocab_size = logits.shape
            logits = logits.view(batch * seq_len, vocab_size)
            target = target.view(batch * seq_len)
            loss = F.cross_entropy(logits, target)

        return logits, loss

    def generate(self, x, max_new_tokens=100):
        # list of tokens
        for _ in range(max_new_tokens):
            x = x[:,-self.config.max_seq_len:] # B, T
            # forward pass
            logits, _ = self(x)
            # only last token matters
            logits = logits[:,-1,:]
            # probablity of all the tokens
            probs = F.softmax(logits, dim=-1)
            # temp?
            # token index
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            x = torch.cat((x, idx_next), dim=1) # (B, T+1)
        return x
        








