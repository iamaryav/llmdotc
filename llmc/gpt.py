import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect
import math
import glob
import numpy as np
import os
from torch.distributed import init_process_group
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch._inductor.config as inductor_config
from contextlib import nullcontext


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


    def forward(self, x, cos, sin, kv_cache=False):
        # x -> (batch, seq_len, hidden_size)
        batch, seq_len = x.shape[:-1]
        # (batch, num_attention_heads, seq_len, head_dim)
        q = self.q_proj(x).view(batch, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if kv_cache:
            if self.cached_key is None:
                k = apply_rotary_emb(k, cos, sin, start_pos=0)
                self.cached_key = k
                self.cached_val = v
                self.cache_seq_len = seq_len
                q_start_pos = 0
            else:
                k = apply_rotary_emb(k, cos, sin, start_pos=self.cache_seq_len)
                q_start_pos = self.cache_seq_len
                k = torch.cat((self.cached_key, k), dim=2)
                v = torch.cat((self.cached_val, v), dim=2)
                self.cached_key = k[:,:,:, -self.sliding_window:]
                self.cached_val = v[:,:,:, -self.sliding_window:]
                self.cache_seq_len = self.cached_key.shape[2]
                k = self.cached_key
                v = self.cached_val
        else:
            k = apply_rotary_emb(k, cos, sin, start_pos=0)
            q_start_pos = 0

        repeat_factor = self.num_attention_heads // self.num_kv_heads
        k = k.repeat_interleave(repeat_factor, dim=1)
        v = v.repeat_interleave(repeat_factor, dim=1)

        q = apply_rotary_emb(q, cos, sin, start_pos=q_start_pos)

        q = norm(q)
        k = norm(k)

        attn_wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        q_len, kv_len = q.shape[2], k.shape[2]
        causal_mask = torch.tril(torch.ones(q_len, kv_len, device=q.device, dtype=torch.bool))
        window_mask = (torch.arange(q_len, device=q.device)[:,None] - torch.arange(kv_len, device=k.device)[None, :]).abs() < self.sliding_window
        causal_mask = causal_mask & window_mask
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
        # residual connection
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
        

    def forward(self, input_ids, target=None, device=None, kv_cache=False):

        batch, seq_len = input_ids.shape # (batch, seq_len)
        tok_emb = self.embed_tokens(input_ids) # (batch, seq_len, hidden_size)

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
        for step in range(max_new_tokens):
            if step == 0:
                cur_token = x[:,-self.config.max_seq_len:]
            else:
                cur_token = idx_next
            logits, _ = self(cur_token, kv_cache=True)
            logits = logits[:,-1,:]
            if temp != 1.0:
                logits = logits / temp
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, idx_next), dim=1)
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

def get_lr(iter):
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
# Distributed data loader
def print0(*args, **kwargs):
    # updating the print statement to print from master gpu - 0
    if int(os.environ.get("RANK", 0)) == 0:
        print(*args, **kwargs)

def _peek_data_shard(filename: str):
    # read the .bin file
    # this information saved during train/val bin file creation
    with open(filename, "rb") as f:
        # read first which 256 int32 value with each size 4 bytes
        header = np.frombuffer(f.read(4 * 256), dtype=np.int32)
        if header[0] != 20240520:
            print("ERROR: Magic number mismatch in the data .bin file")
        assert header[1] == 1, "unsupported version"
        num_token = header[2]
        return num_token

def _load_data_shard(filename: str):
    # read the data from file shard
    with open(filename, "rb") as f:
        # read first which 256 int32 value with each size 4 bytes
        header = np.frombuffer(f.read(4 * 256), np.int32)
        assert header[0] == 20240520, "magic number mismatch in the .bin file"
        assert header[1] == 1, "unsupported file version"
        num_tokens = header[2]
        # the rest of the content file is token
        tokens = np.frombuffer(f.read(), np.uint16)
    assert len(tokens) == num_tokens, "token mismatch from header"
    return tokens

class DistributedDataLoader:
    def __init__(self, filename_pattern, B, T, process_rank, num_processes):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, f"didn't find any files that matches the pattern: {filename_pattern}"
        num_tokens_total = 0
        for file in self.files:
            shard_num_tokens = _peek_data_shard(file)
            assert shard_num_tokens >= num_processes * B * T + 1, f"not enough tokens for 1 batch in the file"
            num_tokens_total += shard_num_tokens
        self.num_tokens_total = num_tokens_total
        print0(f"DataLoader: total number of tokens {num_tokens_total:,} across {len(self.files)} files")

        self.current_shard = None
        self.reset()

    def reset(self):

        if self.current_shard != 0:
            self.current_shard = 0
            self.tokens = _load_data_shard(self.files[self.current_shard])
        self.current_position = self.process_rank * self.B * self.T

    def advance(self): # advance to the next data shard
        self.current_shard = (self.current_shard + 1 ) % len(self.files)
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self):
        B = self.B
        T = self.T
        buff = self.tokens[self.current_position: self.current_position + B * T + 1]
        buff = torch.tensor(buff.astype(np.int32), dtype=torch.long)
        x = (buff[:-1]).view(B, T)
        y = (buff[1:]).view(B, T)
        # next position for this gpu in current shard
        self.current_position += B * T * self.num_processes
        # if not enough token for next batch for all the gpus then move to the next shard
        if self.current_position + ( B * T * self.num_processes + 1) > len(self.tokens):
            self.advance()
        return x, y

# -------------------------------------------
if __name__ == '__main__':
    import time
    import argparse
    import tiktoken
    print0(f"Running pytorch {torch.version.__version__}")

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_bin", type=str, default="dev/data/tinyshakespeare/tiny_shakespeare_val.bin", help="input .bin to train on")
    parser.add_argument("--input_val_bin", type=str, default="", help="input .bin for validation")
    parser.add_argument("--output_dir", type=str, default="", help="output directory for logs")
    parser.add_argument("--model", type=str, default="d12", help="model size: d12/d24/d36/d48")
    parser.add_argument("--num_iterations", type=int, default=100, help="number of training iterations")
    parser.add_argument("--total_batch_size", type=int, default=524288, help="total desired batch size")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16", "float16"], help="precision dtype")
    parser.add_argument("--tensorcores", type=int, default=1, choices=[0, 1], help="enable TF32 precision")
    parser.add_argument("--flash", type=int, default=0, choices=[0, 1], help="use flash attention")
    parser.add_argument("--device", type=str, default="", help="device to use (empty=auto)")
    parser.add_argument("--zero_stage", type=int, default=0, help="ZeRO optimization stage")
    parser.add_argument("--val_loss_every", type=int, default=0, help="validate every N steps (0=off)")
    parser.add_argument("--val_max_steps", type=int, default=20, help="max validation steps")
    parser.add_argument("--sample_every", type=int, default=0, help="sample every N steps (0=off)")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="gradient clipping norm")
    parser.add_argument("--overfit_single_batch", action="store_true", help="overfit a single batch")
    parser.add_argument("--inference_only", action="store_true", help="skip backward pass")
    args = parser.parse_args()
    # training loop
    # just train the model from scratch
    # dataset -> convert it tokens -> test/val split
    # dataloader -> training loop -> forward -> backward -> logging
    # loss curve, lr multiplier, optimizer setup
    # variable override

    # ddp training
    # ddp variable is set by torch
    ddp = int(os.environ.get("RANK", -1)) != -1 # is this a ddp run
    if ddp:
        assert torch.cuda.is_available(), "we need cuda for DDP"
        init_process_group(backend= "nccl")
        ddp_rank = int(os.environ.get("RANK"))
        ddp_local_rank = int(os.environ.get("LOCAL_RANK"))
        ddp_world_size = int(os.environ.get("WORLD_SIZE"))
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0 # we will use master process for logging and checkpointing
        seed_offset = 0
        zero_stage = args.zero_stage
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        seed_offset = 0
        if args.device:
            device = args.device
        else:
            device = "cpu"
            # auto detecting the device
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
    print(f"using device: {device}")
    device_type = "cuda" if "cuda" in device else "cpu"

    B = 8
    T = 2048
    tokens_per_fwdbwd = B * T * ddp_world_size
    assert args.total_batch_size % tokens_per_fwdbwd == 0
    grad_accum_steps = args.total_batch_size // tokens_per_fwdbwd
    print0(f"total desired batch size: {args.total_batch_size}")
    print0(f"=> calculated grad accum steps: {grad_accum_steps}")

    # setup the context manager following the desired data type and device
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()

    # rng
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # set the torch precision mode to use TensorFloat32(TF32) for matmuls
    if args.tensorcores:
        torch.set_float32_matmul_precision("high")

    assert args.flash in {0, 1}

    # init the titokenizer
    enc = tiktoken.get_encoding("gpt2")

    # model initialization from scratch using command line args
    # d with number of layers
    if args.model[0] == "d":
        # from scratch (random weights)
        model_config = {
            "d12": {"max_seq_len": 1024, "vocab_size": 50257, "num_hidden_layers": 12, "num_attention_heads": 12, "hidden_size": 768},
            "d24": {"max_seq_len": 1024, "vocab_size": 50257, "num_hidden_layers": 24, "num_attention_heads": 16, "hidden_size": 1024},
            "d36": {"max_seq_len": 1024, "vocab_size": 50257, "num_hidden_layers": 36, "num_attention_heads": 20, "hidden_size": 1280},
            "d48": {"max_seq_len": 1024, "vocab_size": 50257, "num_hidden_layers": 48, "num_attention_heads": 25, "num_kv_heads": 5, "hidden_size": 1600},
        }[args.model]
        model = GPT(GPTConfig(**model_config))
    model.train()
    model.to(device)
    if args.model:
        if hasattr(inductor_config, "coordinate_descent_tuning"):
            inductor_config.coordinate_descent_tuning = True
        print0(f"compiling the model...")
        model = torch.compile(model)
    #----------------------------------------------------------------------------
    # load tokens using DistributedDateLoader
    train_loader = DistributedDataLoader(args.input_bin, B, T, ddp_rank, ddp_world_size)
    val_loader = None
    if args.input_val_bin:
        val_loader = DistributedDataLoader(args.input_val_bin, B, T, ddp_rank, ddp_world_size)
    #----------------------------------------------------------------------------

    # will write code to write checkpoint and other things in bin format
    # latert for now ddp is focus

    weight_decay = 1e-1
    beta1 = 0.9
    beta2 = 0.95
    if ddp: 
        model = DDP(model, device_ids=[ddp_local_rank])
    optimizer = model.setup_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

    # creating logging directory if it doesn't exist
    logfile = None
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        logfile = os.path.join(args.output_dir, "main.log")
        with open(logfile, "w") as f:
            pass

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    timings = []
    norms = -1.0 # dummy value to print in inference-only
    # now work on the main loop

    #--------------------------------------------------------
    for step in range(args.num_iterations + 1):
        t0 = time.time()
        last_step = (step == args.num_iterations)

        # evaluate the model on validation dataset
        if (args.val_loss_every > 0 \
            and (step % args.val_loss_every == 0 or last_step)) \
            and (val_loader is not None):
            model.eval() 
            val_loader.reset() # to do validation against same set of data every time
            with torch.no_grad():
                val_loss = 0.0
                for _ in range(args.val_max_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    _, loss = model(x, y)
                    val_loss += loss.item()
                val_loss /= args.val_max_steps
            # log to console and to file
            print0(f"val loss {val_loss}")
            if master_process and logfile is not None:
                with open(logfile, "a") as f:
                    f.write("s:%d tel:%f\n" % (step, val_loss))
        # model inference on the master process
        if (args.sample_every > 0 \
            and (step % args.sample_every == 0 or last_step)) \
            and master_process:
            model.eval()
            # to mark start of the output sequence we use "<|endoftext|> token
            start_ids = [enc.eot_token]
            xg = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
            max_new_tokens = 32
            temperature = 1.0
            yg = model.generate(xg, max_new_tokens, temperature=temperature)
            print0('--------------------')
            print0(enc.decode(yg[0].tolist()))
            print0('--------------------')

        # break here so last step validation and inference can happen
        if last_step:
            break
        
        # ---------------------- Training Section Begin ------------------
        model.train()
        optimizer.zero_grad(set_to_none=True)
        # if overfitting a single batch, reset the loader here
        if args.overfit_single_batch:
            train_loader.reset()

        # micro batch where we do gradient accumulation to reach desired total batch size
        lossf = 0.0 # mean loss over the accumulation steps
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            if ddp:
                # hack toggle variable other the official way to do is# with model.no_sync()
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
            # forward pass
            with ctx:
                _, loss = model(x, y)
                loss = loss / grad_accum_steps
                lossf += loss.detach()
            # backward pass
            if not args.inference_only:
                loss.backward()

        if ddp:
            dist.all_reduce(lossf, op=dist.ReduceOp.AVG)
        lossf = lossf.item()
        norm = torch.nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
        # learning rate for this iteration
        lrm = get_lr(step)
        for group in optimizer.param_groups:
            group['lr'] = lrm
        # step the optimizer
        optimizer.step()
        # ---------------------- Training Section End ------------------

        # diagnostics, prints, logging
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()

        # skip 0th iteration for logging
        tokens_per_second = grad_accum_steps * ddp_world_size * B * T / (t1 - t0)
        print0(f"step {step+1:4d}/{args.num_iterations} | train loss {lossf:.6f} | norm {norm:.4f} | lr {lrm:.2e} | ({(t1-t0)*1000:.2f} ms | {tokens_per_second:.0f} tok/s)")

        # log to log file
        if master_process and logfile is not None:
            with open(logfile, "a") as f:
                f.write("s:%d trl:%f\n" % (step, lossf))


        # keep track of the smooth timings, last 20 iterations
        if step > 0 and step > args.num_iterations - 20:
            timings.append(t1-t0)

    # prin the average of the last 20 timings, to get soomth time
    timings = timings[-20:]
    print0(f"final {len(timings)} iters avg: {np.mean(timings)*1000:.3f}ms")
    print0(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")

    # clean up
    if ddp:
        torch.distributed.destroy_process_group()


