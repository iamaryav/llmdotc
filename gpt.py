import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect


# Training pipeline

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
    sliding_window: int = max_seq_len // 4
    # window_pattern: str = "SSSL"

def norm(x):
    return F.rms_norm(x, (x.size(-1),))

def apply_rotary_emb(x, cos, sin, start_pos=0):
    # x1 * cos - x2 * sin, x1 sin + x2 cos
    # embedding -> Q/K Proj -> RoPE rotation -> attention
    # (batch, num_attention_heads, seq_len, head_dim)
    assert x.ndim == 4
    d = x.shape[3] // 2

    # half split
    x1, x2 = x[..., :d], x[..., d:]
    seq_len = x.shape[2]
    cos, sin = cos[:,:,start_pos:start_pos+seq_len], sin[:,:,start_pos:start_pos+seq_len]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos

    return torch.cat([y1, y2], dim=-1)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cached_key = None
        self.cached_val = None
        self.cache_seq_len = 0
        assert config.hidden_size % config.num_attention_heads == 0
        assert config.num_attention_heads % config.num_kv_heads == 0
        self.num_attention_heads = config.num_attention_heads
        self.num_kv_heads = config.num_kv_heads
        self.sliding_window = config.sliding_window
        self.head_dim = config.hidden_size // self.num_attention_heads
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, config.bias)
        self.k_proj = nn.Linear(config.hidden_size, config.num_kv_heads * self.head_dim, config.bias)
        self.v_proj = nn.Linear(config.hidden_size, config.num_kv_heads * self.head_dim, config.bias)
        self.out_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.register_buffer('tril', torch.tril(torch.ones(config.max_seq_len, config.max_seq_len)))


    def forward(self, x, cos, sin, kv_cache=False):
        # x -> (batch, max_seq_len, hidden_size)
        batch, seq_len = x.shape[:-1]
        # (batch, num_attention_heads, seq_len, head_dim)
        q = self.q_proj(x).view(batch, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if kv_cache:
            if self.cached_key is None:
                self.cached_key = k
                self.cached_val = v
                self.cache_seq_len = seq_len
            else:
                k = torch.cat((self.cached_key, k), dim=2)
                v = torch.cat((self.cached_val, v), dim=2)
                self.cached_key = k[:,:,:, -self.sliding_window:]
                self.cached_val = v[:,:,:, -self.sliding_window:]
                self.cache_seq_len = self.cached_key.shape[2]
                k = self.cached_key
                v = self.cached_val

        start_pos = self.cache_seq_len if kv_cache and self.cached_key is not None else 0

        repeat_factor = self.num_attention_heads // self.num_kv_heads
        k = k.repeat_interleave(repeat_factor, dim=1)
        v = v.repeat_interleave(repeat_factor, dim=1)

        q = apply_rotary_emb(q, cos, sin, start_pos=0)
        k = apply_rotary_emb(k, cos, sin, start_pos=start_pos)

        q = norm(q)
        k = norm(k)

        attn_wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        q_len, kv_len = q.shape[2], k.shape[2]
        causal_mask = torch.tril(torch.ones(q_len, kv_len, device=q.device, dtype=torch.bool))
        window_mask = (torch.arange(q_len, device=q.device)[:,None] - torch.arange(kv_len, device=k.device)[None, :]).abs() < self.sliding_window
        causal_mask = causal_mask & window_mask
        # causal_mask = torch.tril(torch.ones(config.sliding_window, config.sliding_window, device=q.device, dtype=torch.bool))
        attn_wei = attn_wei.masked_fill(~causal_mask, float("-inf"))
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

    def forward(self, x, cos, sin, kv_cache=False):
        # pre-attention layernorm -> self attention -> post-attention layernorm -> ffn
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, kv_cache=kv_cache)
        x = x + self.ffn(self.post_attention_norm(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([DecoderLayer(config) for layer_idx in range(config.num_hidden_layers)])
        # self.norm = nn.LayerNorm(config.hidden_size) 
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, config.bias)

        # init all weights
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("down_proj"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.num_hidden_layers))

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

    def _init_weights(self, module):
        # Linear, bias, Embeddings
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def setup_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # AdamW implementation
        # beta 1, beta2, Learning rate, lambda
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f" num decayed parameter tensors: {len(decay_params)} with {num_decay_params}")
        print(f" num non decayed parameter tensors: {len(nodecay_params)} with {num_nodecay_params}")
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        
        return optimizer
    
    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.embed_tokens.weight.numel()
        return n_params
    
    def estimate_mfu(self, fwd_bwd_per_iter, dt):
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.hidden_size // cfg.num_attention_heads, cfg.max_seq_len
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwd_bwd = flops_per_token * T
        flops_per_iter = flops_per_fwd_bwd * fwd_bwd_per_iter
        flops_achieved = flops_per_iter * (1.0/dt)
        flops_promised = 312e12 # GPU's TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu
        

    def forward(self, input_ids, target=None, device='cpu', kv_cache=False):

        batch, seq_len = input_ids.shape
        tok_emb = self.embed_tokens(input_ids)

        # precompute rotary embeddings
        head_dim = self.config.hidden_size // self.config.num_attention_heads

        cos, sin = self._precompute_rotary_embeddings(self.config.max_seq_len, head_dim, device=device)

        x = tok_emb
        for layer in self.layers:
            x = layer(x, cos, sin, kv_cache=kv_cache)
        x = norm(x)
        logits = self.lm_head(x)

        if target == None:
            loss = None
        else:
            batch, seq_len, vocab_size = logits.shape
            logits = logits.view(batch * seq_len, vocab_size)
            target = target.view(batch * seq_len)
            loss = F.cross_entropy(logits, target)

        return logits, loss

    def generate(self, x, max_new_tokens=100, temp=1.0):
        # list of tokens
        for _ in range(max_new_tokens):
            cur_token = x[:,-self.config.max_seq_len:] # (B, T)
            # forward pass
            logits, _ = self(cur_token, kv_cache=True)# (B, T, vocab_size)
            # only last token matters
            logits = logits[:,-1,:] # (B, 1, vocab_size)
            # probablity of all the tokens
            if temp != 1.0:
                logits = logits / temp

            probs = F.softmax(logits, dim=-1) # (B, 1, vocab_size)
            # token index
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            cur_token = idx_next
            x = torch.cat((x, idx_next), dim=1) # (B, T+1)
        return x

# -------------------------------------------
# Other methods
# Like data loaders
# tokenizers
# save params/grads/files to .bin files
# optimizers, and lr multipliers
# learning rate decay scheduler (cosine with warmup)

decay_lr = True # whether to decay the learning rate
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
learning_rate = 6e-4 # max learning rate

def get_lr_multiplier(iter):
    # Linear warmup with cosine decay
    # 1) Linear warmup for warmup iters
    if iter < warmup_iters:
        return learning_rate * (iter + 1) / (warmup_iters + 1)
    # 2) at last min learning rate to avoid zero in the end
    if iter > lr_decay_iters:
        return min_lr
    # 3) cosine decay up to min_lr
    decay_ratio = (iter - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # to coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

device = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------------------------------------
# need to implement better data loader
# there will be list of files - file shard
# with metadata in header with some info like 
# number of tokens

def _peek_data_shard(filename: str):
    # read the .bin file
    # this information saved during train/val bin file creation
    path = ""
    with open(path, "rb") as f:
        # read first 1024 bytes
        header = np.formbuffer(f.read(4 * 256), dtype=np.int32)
        if header[0] != 20240520:
            print("ERROR: Magic number mismatch in the data .bin file")
        assert header[1] == 1, "unsupported version"
        num_token = header[2]
        return num_token
def _load_data_shard(filename: str):

    pass


# --------------------------------------------------------------
batch_size = 8
#  A tiny data loader
data = "shakespeare.txt"
data_dir = os.path.join("data", data)
def get_batch(split):
    if split == "train":
        data = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - max_seq_len, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+config.max_seq_len]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1:i + config.max_seq_len + 1]).astype(np.int64)) for i in ix])
    if device == "cuda":
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)

    return x, y


# -------------------------------------------
if __name__ == '__main__':
    # trining loop
    # just train the model from scratch
    # dataset -> convert it tokens -> test/val split
    # dataloader -> training loop -> forward -> backward -> logging
    # loss curve, lr multiplier, optimizer setup
    # ddp training
    num_iterations = 100
    # math needs to be done
    grad_accum_steps = 8
    weight_decay = 1e-1
    beta1 = 0.9
    beta2 = 0.95
    grad_clip = 1.0
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    vocab_size = 50304
    hidden_size = 768
    intermediate_size = 768 * 4
    max_seq_len = 2048
    num_hidden_layers = 12
    num_attention_heads = 12
    num_kv_heads = 2
    dropout = 0.2
    bias = False
    sliding_window = max_seq_len // 4
    model_args = dict(vocab_size=vocab_size, hidden_size=hidden_size, intermediate_size=intermediate_size, max_seq_len=max_seq_len,                 num_hidden_layers=num_hidden_layers, num_attention_heads=num_attention_heads, num_kv_heads=num_kv_heads, dropout=dropout)
    config = GPTConfig(**model_args)
    model = GPT(config)
    optimizers = model.setup_optimizer(weight_decay, learning_rate, (beta1, beta2), device_type)
    x, y = get_batch("train")
    for step in range(num_iterations + 1):

        # ---------------------------------------------------
        # Evaluate the loss and save the checkpoints
        # if this iteration is the iteration 
        # logs
        # wandb savings
        # save the checkpoint

        # ---------------------------------------------------
        # single training step
        for micro_step in range(grad_accum_steps):
            logits, loss = model(x, y)
            loss = loss / grad_accum_steps
            loss.backward()
            x, y = get_batch('train')
        
        # optimizers
        # gradient clipping 
        if grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        lrm = get_lr_multiplier(step)
        for group in optimizers.param_groups:
            group["lr"] = learning_rate * lrm
        optimizers.step()
        model.zero_grad(set_to_none=True)

        # ---------------------------------------------------
        # training run logs and timings 

