"""Roboactualizer: joint image/action flow-matching policy on a frozen V-JEPA backbone.

A MoT-style JointDiT denoises future image latents and an action chunk together. The clean
current-frame latent z0 leads the image stream and only attends to itself.
"""

import hashlib
import json
import os

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from .backbone import Backbone
from .modules import FeedForward, RMSNorm, modulate


def _load_latent_norm_stats(path: str, embed_dim: int, backbone_name: str | None):
    """Load per-channel latent mean/std; returns (mean, std, identity meta)."""
    with open(path, "rb") as f:
        raw = f.read()
    stats = json.loads(raw)
    mean = torch.tensor(stats["mean"], dtype=torch.float32)
    std = torch.tensor(stats["std"], dtype=torch.float32)
    if mean.shape != (embed_dim,) or std.shape != (embed_dim,):
        raise ValueError(f"latent norm stats must be [{embed_dim}]: {path}")
    meta = {
        "path": os.path.abspath(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "backbone_model_name": stats.get("backbone_model_name"),
        "embed_dim": int(embed_dim),
        "cache_dir": stats.get("cache_dir"),
        "num_elements": stats.get("num_elements"),
    }
    return mean, std, meta


# ============================================================
# Flow-matching scheduler
# ============================================================

class FlowMatchScheduler:
    """Shifted-sigma flow matching: x-prediction trained with a velocity loss."""

    def __init__(self, num_train_timesteps: int = 1000, shift: float = 5.0, sigma_clip_min: float = 0.05):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_clip_min = sigma_clip_min
        ts = torch.linspace(1.0 / num_train_timesteps, 1.0, num_train_timesteps)
        self.sigmas = self._phi(ts)
        self.timesteps = self.sigmas * num_train_timesteps
        self.inference_sigmas = None
        self.inference_timesteps = None
        self.dt = None

    def _phi(self, u: torch.Tensor) -> torch.Tensor:
        return self.shift * u / (1.0 + (self.shift - 1.0) * u)

    def sample_train(self, batch_size: int, device):
        t_id = torch.randint(0, self.num_train_timesteps, (batch_size,), device=device)
        return self.sigmas.to(device)[t_id].view(-1, 1, 1), self.timesteps.to(device)[t_id]

    def add_noise(self, x_clean, noise, sigma):
        return (1.0 - sigma) * x_clean + sigma * noise

    def velocity_target(self, x_clean, z_noisy, sigma):
        return (z_noisy - x_clean) / sigma.clamp(min=self.sigma_clip_min)

    def x_pred_to_velocity(self, x_pred, z_noisy, sigma):
        return (z_noisy - x_pred) / sigma.clamp(min=self.sigma_clip_min)

    def set_inference_steps(self, num_inference_steps: int):
        ts = torch.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps)
        sigmas = self._phi(ts)
        self.inference_sigmas = sigmas
        self.inference_timesteps = sigmas * self.num_train_timesteps
        next_sigmas = torch.cat([sigmas[1:], torch.zeros(1)])
        self.dt = next_sigmas - sigmas

    def scale_for_model_input(self, t_idx: int, device):
        return self.inference_timesteps.to(device)[t_idx : t_idx + 1]

    def sigma_for_step(self, t_idx: int, device):
        return self.inference_sigmas.to(device)[t_idx : t_idx + 1]


# ============================================================
# Embeddings and RoPE
# ============================================================

def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(position.type(torch.float64), torch.pow(
        10000, -torch.arange(dim//2, dtype=torch.float64, device=position.device).div(dim//2)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                   [: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def rope_apply(x, freqs, num_heads):
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_out = torch.view_as_complex(x.to(torch.float64).reshape(
        x.shape[0], x.shape[1], x.shape[2], -1, 2))
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    half = dim // 2
    h_pairs = w_pairs = half // 3
    f_pairs = half - 2 * h_pairs
    f_freqs_cis = precompute_freqs_cis(2 * f_pairs, end, theta)
    h_freqs_cis = precompute_freqs_cis(2 * h_pairs, end, theta)
    w_freqs_cis = precompute_freqs_cis(2 * w_pairs, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


def build_image_rope_freqs(f_positions, grid_h, grid_w, head_dim):
    """3D RoPE over [frame][h][w]; the frame axis uses real timesteps. -> [F*gh*gw, 1, head_dim//2]."""
    f_freqs, h_freqs, w_freqs = precompute_freqs_cis_3d(head_dim)
    n_frame = f_positions.shape[0]
    freqs = torch.cat([
        f_freqs[f_positions].view(n_frame, 1, 1, -1).expand(n_frame, grid_h, grid_w, -1),
        h_freqs[:grid_h].view(1, grid_h, 1, -1).expand(n_frame, grid_h, grid_w, -1),
        w_freqs[:grid_w].view(1, 1, grid_w, -1).expand(n_frame, grid_h, grid_w, -1),
    ], dim=-1).reshape(n_frame * grid_h * grid_w, 1, -1)
    return freqs


def build_action_rope_freqs(seq_len: int, head_dim: int, theta: float = 10000.0):
    """1D RoPE at positions [0, seq_len). -> [seq_len, 1, head_dim//2]."""
    return precompute_freqs_cis(head_dim, theta=theta)[:seq_len].view(seq_len, 1, -1)


# ============================================================
# Input tokens
# ============================================================

class TimeEmbedder(nn.Module):
    """t -> (time_emb [B, D], adaln [B, 6, D]); one shared adaLN head per branch."""

    def __init__(self, d_model: int, freq_dim: int = 256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.adaln_head = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 6 * d_model),
        )
        nn.init.zeros_(self.adaln_head[-1].weight)
        nn.init.zeros_(self.adaln_head[-1].bias)

    def forward(self, t: torch.Tensor):
        time_emb = self.mlp(sinusoidal_embedding_1d(self.freq_dim, t))
        adaln = self.adaln_head(time_emb).reshape(t.size(0), 6, -1)
        return time_emb, adaln


class ContextEmbedding(nn.Module):
    """Linear -> GELU(tanh) -> Linear."""

    def __init__(self, in_dim: int, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.GELU(approximate="tanh"),
            nn.Linear(d_model, d_model),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return self.net(context)


class ContextTokenBuilder(nn.Module):
    """Cross-attn context: text tokens plus one trailing proprio token."""

    def __init__(self, text_dim: int, proprio_dim: int, d_model: int):
        super().__init__()
        self.text_embedding = ContextEmbedding(text_dim, d_model)
        self.proprio_embedding = ContextEmbedding(proprio_dim, d_model)

    def forward(self, text_context, text_mask, proprio):
        text_tok = self.text_embedding(text_context)  # [B, L, D]
        prop_tok = self.proprio_embedding(proprio).unsqueeze(1)  # [B, 1, D]
        context = torch.cat([text_tok, prop_tok], dim=1)
        context_mask = torch.cat([text_mask, text_mask.new_ones(text_mask.size(0), 1)], dim=1)
        return context, context_mask


class ImageTokenEncoder(nn.Module):
    """Shared latent projection for z0 and future image latents."""

    def __init__(self, d_backbone: int, d_model: int, num_patches: int, future_size: int):
        super().__init__()
        self.num_patches = num_patches
        self.future_size = future_size
        self.latent_proj = nn.Linear(d_backbone, d_model)

    def forward(self, z0, z_future_noisy):
        prefix = self.latent_proj(z0)  # [B, N, D]
        future = self.latent_proj(z_future_noisy)  # [B, F, N, D]
        return prefix, future.reshape(future.size(0), self.future_size * self.num_patches, -1)


class ActionTokenEncoder(nn.Module):
    def __init__(self, d_model: int, action_dim: int):
        super().__init__()
        self.action_encoder = nn.Linear(action_dim, d_model)

    def forward(self, noisy_actions: torch.Tensor) -> torch.Tensor:
        return self.action_encoder(noisy_actions)


# ============================================================
# JointDiT
# ============================================================

class CrossAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, dropout: float = 0.0):
        super().__init__()
        inner = heads * dim_head
        self.heads = heads
        self.to_q = nn.Linear(dim, inner, bias=False)
        self.to_kv = nn.Linear(dim, inner * 2, bias=False)
        self.norm_q = RMSNorm(inner, eps=1e-6)
        self.norm_k = RMSNorm(inner, eps=1e-6)
        self.to_out = nn.Sequential(nn.Linear(inner, dim), nn.Dropout(dropout))
        self.dropout = dropout

    def context_kv(self, context):
        """Query-independent (normed K, raw KV) for the cross-attn KV cache."""
        kv = self.to_kv(context)
        return self.norm_k(kv.chunk(2, dim=-1)[0]), kv

    def forward(self, x, context, context_mask, context_kv=None):
        q = self.norm_q(self.to_q(x))
        if context_kv is None:
            k, v = self.to_kv(context).chunk(2, dim=-1)
            k = self.norm_k(k)
        else:
            k, kv = context_kv
            v = kv.chunk(2, dim=-1)[1]
        q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=self.heads) for t in (q, k, v))
        drop = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=context_mask[:, None, None, :], dropout_p=drop, is_causal=False
        )
        return self.to_out(rearrange(out, "b h n d -> b n (h d)"))


class BranchSelfAttention(nn.Module):
    """Per-branch q/k/v and output projections; attention itself runs jointly in JointDiT."""

    def __init__(self, dim: int, heads: int, dim_head: int):
        super().__init__()
        inner = heads * dim_head
        self.heads = heads
        self.to_qkv = nn.Linear(dim, inner * 3, bias=False)
        self.norm_q = RMSNorm(inner, eps=1e-6)
        self.norm_k = RMSNorm(inner, eps=1e-6)
        self.to_out = nn.Linear(inner, dim)

    def qkv(self, x_normed):
        q, k, v = self.to_qkv(x_normed).chunk(3, dim=-1)
        return self.norm_q(q), self.norm_k(k), v


def build_mixed_attn_mask(n_z0, n_future, n_action, device):
    """Self-attn visibility over [z0 | future image | action]; True = visible.

    z0 sees only itself; future image and action see z0 and their own branch.
    """
    n_image = n_z0 + n_future
    mask = torch.ones(n_image + n_action, n_image + n_action, dtype=torch.bool, device=device)
    z0, fut, act = slice(0, n_z0), slice(n_z0, n_image), slice(n_image, None)
    mask[z0, fut] = False
    mask[z0, act] = False
    mask[fut, act] = False
    mask[act, fut] = False
    return mask


def attend(q, k, v, heads):
    """Unmasked SDPA for the cached paths.

    Pinned to EFFICIENT: the masked reference path uses it, FLASH would differ by 1 ULP.
    """
    q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=heads) for t in (q, k, v))
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    return rearrange(out, "b h n d -> b n (h d)")


def as_joint_slice(out, n_before, n_after):
    """View ``out`` as a slice of a [B, n_before + N + n_after, C] buffer, like the joint forward's split.

    At B > 1 such a slice makes F.linear skip the fused-bias GEMM; matching it keeps cached paths bit-exact.
    """
    b, n, c = out.shape
    buf = out.new_empty(b, n_before + n + n_after, c)
    buf[:, n_before:n_before + n] = out
    return buf[:, n_before:n_before + n]


def mot_self_attention(q_image, q_action, k_list, v_list, heads, n_image, attn_mask):
    """One SDPA over both branches' concatenated tokens, split back per branch."""
    q = torch.cat([q_image, q_action], dim=1)
    k = torch.cat(k_list, dim=1)
    v = torch.cat(v_list, dim=1)
    q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=heads) for t in (q, k, v))
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False)
    out = rearrange(out, "b h n d -> b n (h d)")
    return out[:, :n_image], out[:, n_image:]


class JointBranchBlock(nn.Module):
    """One branch of a JointDiT layer, split around the shared self-attention."""

    def __init__(self, d_model, heads, dim_head, mlp_dim, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.norm3 = nn.LayerNorm(d_model, eps=1e-6)
        self.self_attn = BranchSelfAttention(d_model, heads, dim_head)
        self.cross_attn = CrossAttention(d_model, heads, dim_head, dropout)
        self.ffn = FeedForward(d_model, mlp_dim)
        self.mod = nn.Parameter(torch.zeros(6, d_model))

    def attn_in(self, x, adaln_params):
        """AdaLN-modulated q/k/v; ``adaln_params`` is [B, T, 6, D] (T=1 broadcasts)."""
        (msa_shift, msa_scale, msa_gate,
         ffn_shift, ffn_scale, ffn_gate) = (adaln_params + self.mod).unbind(dim=2)
        q, k, v = self.self_attn.qkv(modulate(self.norm1(x), msa_shift, msa_scale))
        return q, k, v, msa_gate, (ffn_shift, ffn_scale, ffn_gate)

    def adaln_mods(self, adaln_params):
        """t-only part of ``attn_in`` with scales pre-added by 1 (for the AdaLN cache)."""
        shift, scale, gate, f_shift, f_scale, f_gate = (adaln_params + self.mod).unbind(dim=2)
        return shift, 1 + scale, gate, f_shift, 1 + f_scale, f_gate

    def attn_in_mods(self, x, mods):
        """``attn_in`` from precomputed ``adaln_mods``."""
        shift, scale1, gate, f_shift, f_scale1, f_gate = mods
        q, k, v = self.self_attn.qkv(self.norm1(x) * scale1 + shift)
        return q, k, v, gate, (f_shift, f_scale1, f_gate)

    def post_mot(self, x, mixed, msa_gate, ffn_params, context, context_mask,
                 scale_plus_one: bool = False, context_kv=None):
        ffn_shift, ffn_scale, ffn_gate = ffn_params
        x = x + msa_gate * self.self_attn.to_out(mixed)
        x = x + self.cross_attn(self.norm3(x), context, context_mask, context_kv)
        h = self.norm2(x)
        h = h * ffn_scale + ffn_shift if scale_plus_one else modulate(h, ffn_shift, ffn_scale)
        return x + ffn_gate * self.ffn(h)


class JointDiT(nn.Module):
    """N MoT layers; each updates the image (z0 + future) and action branches."""

    def __init__(self, num_layers, d_model, heads, dim_head, mlp_dim, dropout: float = 0.0):
        super().__init__()
        self.heads = heads
        self.image_blocks = nn.ModuleList(
            [JointBranchBlock(d_model, heads, dim_head, mlp_dim, dropout) for _ in range(num_layers)]
        )
        self.action_blocks = nn.ModuleList(
            [JointBranchBlock(d_model, heads, dim_head, mlp_dim, dropout) for _ in range(num_layers)]
        )

    def forward(self, image_state, action, image_context, action_context, context_mask,
                image_adaln, action_adaln, n_z0, image_freqs, action_freqs, action_context_mask=None):
        """``action_context_mask`` differs from ``context_mask`` only under CFG."""
        if action_context_mask is None:
            action_context_mask = context_mask
        n_image = image_state.size(1)
        attn_mask = build_mixed_attn_mask(n_z0, n_image - n_z0, action.size(1), image_state.device)
        image_freqs = image_freqs.to(image_state.device)
        action_freqs = action_freqs.to(action.device)
        for image_blk, act_blk in zip(self.image_blocks, self.action_blocks):
            q_i, k_i, v_i, gate_i, ffn_i = image_blk.attn_in(image_state, image_adaln)
            q_a, k_a, v_a, gate_a, ffn_a = act_blk.attn_in(action, action_adaln)
            q_i = rope_apply(q_i, image_freqs, self.heads)
            k_i = rope_apply(k_i, image_freqs, self.heads)
            q_a = rope_apply(q_a, action_freqs, self.heads)
            k_a = rope_apply(k_a, action_freqs, self.heads)
            mixed_i, mixed_a = mot_self_attention(
                q_i, q_a, [k_i, k_a], [v_i, v_a], self.heads, n_image, attn_mask
            )
            image_state = image_blk.post_mot(image_state, mixed_i, gate_i, ffn_i, image_context, context_mask)
            action = act_blk.post_mot(action, mixed_a, gate_a, ffn_a, action_context, action_context_mask)
        return image_state, action

    def prefill_z0_cache(self, image_state, image_context, context_mask, image_adaln, image_freqs, n_action,
                         mods=None):
        """Run z0 through the image branch once and cache each layer's post-RoPE K/V."""
        image_freqs = image_freqs.to(image_state.device)
        cache = []
        for idx, image_blk in enumerate(self.image_blocks):
            if mods is None:
                q_i, k_i, v_i, gate_i, ffn_i = image_blk.attn_in(image_state, image_adaln)
            else:
                q_i, k_i, v_i, gate_i, ffn_i = image_blk.attn_in_mods(image_state, mods[idx])
            q_i = rope_apply(q_i, image_freqs, self.heads)
            k_i = rope_apply(k_i, image_freqs, self.heads)
            mixed_i = as_joint_slice(attend(q_i, k_i, v_i, self.heads), 0, n_action)
            image_state = image_blk.post_mot(
                image_state, mixed_i, gate_i, ffn_i, image_context, context_mask, scale_plus_one=mods is not None,
            )
            cache.append((k_i, v_i))
        return cache

    def forward_action_with_cache(self, action, action_context, context_mask, action_adaln,
                                  action_freqs, z0_cache, mods=None, cross_kv=None):
        """Action branch only, attending to cached z0 K/V (action rows need no mask)."""
        action_freqs = action_freqs.to(action.device)
        for idx, (act_blk, (k_i, v_i)) in enumerate(zip(self.action_blocks, z0_cache)):
            if mods is None:
                q_a, k_a, v_a, gate_a, ffn_a = act_blk.attn_in(action, action_adaln)
            else:
                q_a, k_a, v_a, gate_a, ffn_a = act_blk.attn_in_mods(action, mods[idx])
            q_a = rope_apply(q_a, action_freqs, self.heads)
            k_a = rope_apply(k_a, action_freqs, self.heads)
            mixed_a = attend(q_a, torch.cat([k_i, k_a], dim=1), torch.cat([v_i, v_a], dim=1), self.heads)
            mixed_a = as_joint_slice(mixed_a, k_i.size(1), 0)
            action = act_blk.post_mot(
                action, mixed_a, gate_a, ffn_a, action_context, context_mask,
                scale_plus_one=mods is not None, context_kv=None if cross_kv is None else cross_kv[idx],
            )
        return action


class ActionDecoder(nn.Module):
    """Zero-initialized linear head."""

    def __init__(self, d_model: int, action_dim: int):
        super().__init__()
        self.head = nn.Linear(d_model, action_dim)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, a_final: torch.Tensor) -> torch.Tensor:
        return self.head(a_final)


class ImageDecoder(nn.Module):
    """LayerNorm -> time-conditioned modulation -> zero-initialized linear."""

    def __init__(self, d_model: int, latent_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.modulation = nn.Parameter(torch.randn(2, d_model) / d_model**0.5)
        self.head = nn.Linear(d_model, latent_dim)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, image_future_final, time_emb, future_size: int, num_patches: int):
        params = self.modulation.unsqueeze(0) + time_emb.unsqueeze(1).expand(-1, 2, -1)
        shift, scale = (p.unsqueeze(1) for p in params.unbind(dim=1))
        x = self.head(modulate(self.norm(image_future_final), shift, scale))
        return x.reshape(x.size(0), future_size, num_patches, -1)


# ============================================================
# Policy
# ============================================================

class CudaGraphFn:
    """Capture ``fn(*args, *consts)`` as a CUDA graph; args are copied in per call."""

    def __init__(self, fn, args, consts=()):
        self.args = [a.clone() for a in args]
        self.consts = [c.clone() for c in consts]

        def run():
            return fn(*self.args, *self.consts)

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):  # warm up lazy kernels outside the graph
                run()
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.out = run()

    def update_consts(self, consts):
        for dst, src in zip(self.consts, consts):
            dst.copy_(src)

    def __call__(self, *args):
        for dst, src in zip(self.args, args):
            dst.copy_(src)
        self.graph.replay()
        return self.out


class FlowPolicy(nn.Module):
    """Frozen backbone + text/proprio context + JointDiT + image/action decoders."""

    def __init__(
        self,
        backbone: Backbone,
        *,
        chunk_size: int,
        future_size: int,
        video_freq_ratio: int,
        action_dim: int,
        text_dim: int,
        proprio_dim: int,
        d_model: int = 384,
        num_layers: int = 12,
        num_heads: int = 6,
        dim_head: int = 64,
        mlp_dim: int = 1536,
        dropout: float = 0.0,
        freq_dim: int = 256,
        num_train_timesteps: int = 1000,
        shift: float = 5.0,
        num_inference_steps: int = 10,
        action_weight: float = 1.0,
        image_predict_weight: float = 1.0,
        rgb_latent_dtype: str = "float16",
        latent_norm_stats_path: str | None = None,
        eval_encode_fp32: bool = False,
    ):
        super().__init__()
        self.backbone = backbone
        self.backbone.eval()
        self.backbone.requires_grad_(False)

        # Per-channel latent norm; stats come from a file when training, from the ckpt otherwise.
        self.latent_norm_stats_meta = None
        if latent_norm_stats_path:
            ln_mean, ln_std, self.latent_norm_stats_meta = _load_latent_norm_stats(
                str(latent_norm_stats_path), int(backbone.embed_dim), getattr(backbone, "model_name", None)
            )
        else:
            ln_mean = torch.zeros(int(backbone.embed_dim), dtype=torch.float32)
            ln_std = torch.ones(int(backbone.embed_dim), dtype=torch.float32)
        self.register_buffer("latent_norm_mean", ln_mean)
        self.register_buffer("latent_norm_std", ln_std)

        self.chunk_size = chunk_size
        self.future_size = future_size
        self.video_freq_ratio = int(video_freq_ratio)
        self.action_dim = action_dim
        self.num_patches = backbone.num_patches
        self.num_inference_steps = num_inference_steps
        self.action_weight = action_weight
        self.image_predict_weight = image_predict_weight
        self.rgb_latent_dtype = {"float16": torch.float16, "float32": torch.float32}[str(rgb_latent_dtype)]
        self.d_model = d_model
        # Encode z0 in fp32 at eval, matching the precomputed training latents.
        self.eval_encode_fp32 = bool(eval_encode_fp32)
        self.metadata = {}

        self.image_context_builder = ContextTokenBuilder(text_dim, proprio_dim, d_model)
        self.action_context_builder = ContextTokenBuilder(text_dim, proprio_dim, d_model)
        self.image_token_encoder = ImageTokenEncoder(backbone.embed_dim, d_model, self.num_patches, future_size)
        self.action_token_encoder = ActionTokenEncoder(d_model, action_dim)
        self.image_time_embedder = TimeEmbedder(d_model, freq_dim=freq_dim)
        self.action_time_embedder = TimeEmbedder(d_model, freq_dim=freq_dim)
        self.joint_expert = JointDiT(num_layers, d_model, num_heads, dim_head, mlp_dim, dropout)
        self.image_decoder = ImageDecoder(d_model, backbone.embed_dim)
        self.action_decoder = ActionDecoder(d_model, action_dim)
        self.image_scheduler = FlowMatchScheduler(num_train_timesteps, shift)
        self.action_scheduler = FlowMatchScheduler(num_train_timesteps, shift)

        # Non-persistent RoPE buffers: static addresses for CUDA graphs, absent from ckpts.
        grid_h, grid_w = backbone.patch_grid
        f_positions = torch.arange(future_size + 1) * video_freq_ratio  # z0 at 0, frame j at j*r
        self.register_buffer(
            "image_rope_freqs", build_image_rope_freqs(f_positions, grid_h, grid_w, dim_head), persistent=False
        )
        self.register_buffer(
            "action_rope_freqs", build_action_rope_freqs(chunk_size, dim_head), persistent=False
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def _normalize_latent(self, z):
        mean = self.latent_norm_mean.to(device=z.device, dtype=z.dtype)
        std = self.latent_norm_std.to(device=z.device, dtype=z.dtype)
        return (z - mean) / std

    def _action_stream(self, a_noisy, t_action):
        """-> (tokens, adaln [B, 1, 6, D], time_emb)."""
        action_time_emb, action_adaln = self.action_time_embedder(t_action)
        return self.action_token_encoder(a_noisy), action_adaln.unsqueeze(1), action_time_emb

    def _joint_forward(self, z0, z_future_noisy, a_noisy, image_context, action_context, context_mask,
                       t_image, t_action):
        """Full forward; returns clean-data predictions (x_image, x_action)."""
        image_time_emb, future_adaln = self.image_time_embedder(t_image)
        _, z0_adaln = self.image_time_embedder(torch.zeros_like(t_image))
        image_prefix, image_future = self.image_token_encoder(z0, z_future_noisy)
        image_state = torch.cat([image_prefix, image_future], dim=1)
        n_z0 = image_prefix.size(1)
        image_adaln = torch.cat([
            z0_adaln.unsqueeze(1).expand(-1, n_z0, -1, -1),
            future_adaln.unsqueeze(1).expand(-1, image_future.size(1), -1, -1),
        ], dim=1)
        action_tokens, action_adaln, _ = self._action_stream(a_noisy, t_action)
        image_out, act_out = self.joint_expert(
            image_state, action_tokens, image_context, action_context, context_mask,
            image_adaln, action_adaln, n_z0, self.image_rope_freqs, self.action_rope_freqs,
        )
        x_image = self.image_decoder(image_out[:, n_z0:], image_time_emb, self.future_size, self.num_patches)
        return x_image, self.action_decoder(act_out)

    def _z0_tokens(self, z0, t_action):
        """z0 -> (tokens, adaln at t=0)."""
        _, z0_adaln = self.image_time_embedder(torch.zeros_like(t_action))
        image_prefix = self.image_token_encoder.latent_proj(z0)
        return image_prefix, z0_adaln.unsqueeze(1).expand(-1, image_prefix.size(1), -1, -1)

    def _action_only_forward(self, z0, a_noisy, image_context, action_context, context_mask, t_action,
                             action_context_mask=None):
        """z0 + action forward without future image tokens."""
        image_prefix, image_adaln = self._z0_tokens(z0, t_action)
        n_z0 = image_prefix.size(1)
        action_tokens, action_adaln, _ = self._action_stream(a_noisy, t_action)
        _, act_out = self.joint_expert(
            image_prefix, action_tokens, image_context, action_context, context_mask,
            image_adaln, action_adaln, n_z0, self.image_rope_freqs[:n_z0], self.action_rope_freqs,
            action_context_mask=action_context_mask,
        )
        return self.action_decoder(act_out)

    def _adaln_tables(self, bsz, device):
        """Per-layer AdaLN mods for z0 (t=0) and every denoising step, cached across calls.

        Keyed on weight versions, autocast state and the timestep schedule.
        """
        sched = self.action_scheduler
        params = self.__dict__.get("_adaln_params")
        if params is None:
            params = self.__dict__["_adaln_params"] = [
                *self.image_time_embedder.parameters(), *self.action_time_embedder.parameters(),
                *(b.mod for b in self.joint_expert.image_blocks), *(b.mod for b in self.joint_expert.action_blocks)]
        key = (bsz, device, torch.is_autocast_enabled(device.type), torch.get_autocast_dtype(device.type),
               sched.inference_timesteps.tolist(), [(p.data_ptr(), p._version) for p in params])
        cached = self.__dict__.get("_adaln_cache")
        if cached is not None and cached[0] == key:
            return cached[1]
        _, z0_adaln = self.image_time_embedder(torch.zeros(bsz, device=device))
        z0_table = torch.stack([torch.stack(b.adaln_mods(z0_adaln.unsqueeze(1))) for b in self.joint_expert.image_blocks])
        action_tables = []
        for i in range(len(sched.inference_timesteps)):
            _, act_adaln = self.action_time_embedder(sched.scale_for_model_input(i, device).expand(bsz))
            action_tables.append(torch.stack([torch.stack(b.adaln_mods(act_adaln.unsqueeze(1)))
                                              for b in self.joint_expert.action_blocks]))
        tables = (z0_table, torch.stack(action_tables))
        self.__dict__["_adaln_cache"] = (key, tables)
        return tables

    def _prefill_z0_cache(self, z0, image_context, context_mask, t_action, mods=None):
        if mods is None:
            image_prefix, image_adaln = self._z0_tokens(z0, t_action)
        else:
            image_prefix, image_adaln = self.image_token_encoder.latent_proj(z0), None
        return self.joint_expert.prefill_z0_cache(
            image_prefix, image_context, context_mask, image_adaln,
            self.image_rope_freqs[: image_prefix.size(1)], self.chunk_size, mods,
        )

    def _action_cross_kv(self, action_context):
        return [blk.cross_attn.context_kv(action_context) for blk in self.joint_expert.action_blocks]

    def _action_only_forward_cached(self, z0_cache, a_noisy, action_context, context_mask, t_action,
                                    mods=None, cross_kv=None):
        """``_action_only_forward`` with cached z0 K/V (and optional AdaLN / cross-attn caches)."""
        if mods is None:
            action_tokens, action_adaln, _ = self._action_stream(a_noisy, t_action)
        else:
            action_tokens, action_adaln = self.action_token_encoder(a_noisy), None
        act_out = self.joint_expert.forward_action_with_cache(
            action_tokens, action_context, context_mask, action_adaln, self.action_rope_freqs,
            z0_cache, mods, cross_kv,
        )
        return self.action_decoder(act_out)

    def _graphed(self, name, fn, args, consts=()):
        """Capture ``fn`` once per (name, shapes, dtypes); refresh consts on reuse."""
        key = tuple((tuple(t.shape), t.dtype) for t in (*args, *consts))
        graphs = self.__dict__.setdefault("_cuda_graphs", {})
        if graphs.get(name, (None,))[0] != key:
            graphs[name] = (key, CudaGraphFn(fn, args, consts))
        graphs[name][1].update_consts(consts)
        return graphs[name][1]

    @staticmethod
    def _masked_mean(loss_token, is_pad):
        if is_pad is None:
            return loss_token.mean(dim=1)
        valid = (~is_pad).to(device=loss_token.device, dtype=loss_token.dtype)
        return (loss_token * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)

    def _encode_window(self, batch):
        """(z0, z_future) from cached latents, or online from pixel clips."""
        if "z0" in batch:
            z0, z_future = batch["z0"], batch["z_future"]
            return self._normalize_latent(z0), self._normalize_latent(z_future)
        b, t, clip_len, c, h, w = batch["pixels"].shape
        clips = batch["pixels"].reshape(b * t, clip_len, c, h, w).float()
        with torch.autocast(device_type=clips.device.type, enabled=False):
            z = self.backbone.encode_clip_last_group(clips)
        z = z.to(self.rgb_latent_dtype).float()  # mirror the cache quantization
        z = self._normalize_latent(z.reshape(b, t, self.num_patches, self.backbone.embed_dim))
        return z[:, 0], z[:, 1 : 1 + self.future_size]

    def training_loss(self, batch):
        """Joint flow-matching loss over the action chunk and future image latents."""
        text_context = batch["text_context"]
        text_mask = batch["text_mask"]
        proprio = batch["proprio"]
        action_chunk = batch["action_chunk"]
        bsz = action_chunk.size(0)
        device = action_chunk.device

        z0, z_future = self._encode_window(batch)
        z_future = z_future.float()
        image_context, context_mask = self.image_context_builder(text_context, text_mask, proprio)
        action_context, _ = self.action_context_builder(text_context, text_mask, proprio)

        sigma_a, t_a = self.action_scheduler.sample_train(bsz, device)
        noise_a = torch.randn_like(action_chunk)
        a_noisy = self.action_scheduler.add_noise(action_chunk, noise_a, sigma_a)
        v_a_target = self.action_scheduler.velocity_target(action_chunk, a_noisy, sigma_a)

        sigma_r, t_r = self.image_scheduler.sample_train(bsz, device)
        sigma_r = sigma_r.unsqueeze(-1)  # [B,1,1,1]
        noise_r = torch.randn_like(z_future)
        z_future_noisy = self.image_scheduler.add_noise(z_future, noise_r, sigma_r)
        v_r_target = self.image_scheduler.velocity_target(z_future, z_future_noisy, sigma_r)

        x_r_hat, x_a_hat = self._joint_forward(
            z0, z_future_noisy, a_noisy, image_context, action_context, context_mask, t_r, t_a
        )
        v_a_hat = self.action_scheduler.x_pred_to_velocity(x_a_hat, a_noisy, sigma_a)
        v_r_hat = self.image_scheduler.x_pred_to_velocity(x_r_hat, z_future_noisy, sigma_r)

        action_loss_token = F.mse_loss(v_a_hat.float(), v_a_target.float(), reduction="none").mean(dim=2)
        action_loss = self._masked_mean(action_loss_token, batch.get("action_is_pad")).mean()
        image_loss_token = F.mse_loss(v_r_hat.float(), v_r_target.float(), reduction="none").mean(dim=(2, 3))
        image_loss = self._masked_mean(image_loss_token, batch["image_is_pad"][:, 1:]).mean()

        loss = self.action_weight * action_loss + self.image_predict_weight * image_loss
        logs = {
            "action_loss": action_loss.detach(),
            "image_loss": image_loss.detach(),
            "action_sigma_mean": sigma_a.mean().detach(),
            "image_sigma_mean": sigma_r.mean().detach(),
        }
        return loss, logs

    @torch.no_grad()
    def generate(
        self,
        obs_pixels,
        text_context,
        text_mask,
        proprio,
        num_inference_steps: int = 10,
        cfg_scale: float = 1.0,
        uncond_text_context=None,
        uncond_text_mask=None,
        use_kv_cache: bool = True,
        use_cuda_graph: bool = True,
        use_adaln_cache: bool = True,
        use_cross_kv_cache: bool = True,
    ):
        """Euler-denoise an action chunk [B, K, A] from noise, conditioned on a z0 clip.

        ``cfg_scale != 1``: action text-CFG; the uncond pass swaps only the action-branch text.
        The four cache flags are pure speedups; the last three require ``use_kv_cache``.
        """
        bsz = obs_pixels.size(0)
        device = obs_pixels.device
        if self.eval_encode_fp32:
            with torch.autocast(device_type=device.type, enabled=False):
                z0 = self.backbone.encode_clip_last_group(obs_pixels.float())
            z0 = z0.to(self.rgb_latent_dtype).float()
        else:
            z0 = self.backbone.encode_clip_last_group(obs_pixels)
        z0 = self._normalize_latent(z0)
        image_context, context_mask = self.image_context_builder(text_context, text_mask, proprio)
        action_context, _ = self.action_context_builder(text_context, text_mask, proprio)

        use_cfg = float(cfg_scale) != 1.0
        if use_cfg:
            # cond | uncond stacked into one 2B forward; only the action context changes.
            u_action_context, u_mask = self.action_context_builder(uncond_text_context, uncond_text_mask, proprio)
            z0 = torch.cat([z0, z0], dim=0)
            image_context = torch.cat([image_context, image_context], dim=0)
            action_context = torch.cat([action_context, u_action_context], dim=0)
            action_context_mask = torch.cat([context_mask, u_mask], dim=0)
            context_mask = torch.cat([context_mask, context_mask], dim=0)
        else:
            action_context_mask = context_mask
        n_fwd = 2 * bsz if use_cfg else bsz

        self.action_scheduler.set_inference_steps(num_inference_steps)
        a_t = torch.randn(bsz, self.chunk_size, self.action_dim, device=device)
        dt_a = self.action_scheduler.dt.to(device)

        z0_cache = graphed = act_tables = cross_kv = None
        if use_kv_cache:
            t0 = torch.zeros(n_fwd, device=device)
            z0_table = None
            if use_adaln_cache:
                z0_table, act_tables = self._adaln_tables(n_fwd, device)
            if not use_cuda_graph:
                z0_cache = self._prefill_z0_cache(z0, image_context, context_mask, t0, z0_table)
                if use_cross_kv_cache:
                    cross_kv = self._action_cross_kv(action_context)
            else:
                if use_adaln_cache:
                    prefill = self._graphed(
                        "prefill_adaln", lambda z, c, m, tab: self._prefill_z0_cache(z, c, m, None, tab),
                        (z0, image_context, context_mask), (z0_table,))
                    z0_cache = prefill(z0, image_context, context_mask)
                else:
                    prefill = self._graphed("prefill", self._prefill_z0_cache, (z0, image_context, context_mask, t0))
                    z0_cache = prefill(z0, image_context, context_mask, t0)
                flat = [t for kv in z0_cache for t in kv]
                n_z0kv = len(flat)
                if use_cross_kv_cache:
                    cross = self._graphed("cross_kv", self._action_cross_kv, (action_context,))(action_context)
                    flat += [t for kv in cross for t in kv]

                def step(a, t, ctx, mask, *kvs):
                    z0c = list(zip(kvs[0:n_z0kv:2], kvs[1:n_z0kv:2]))
                    xkv = list(zip(kvs[n_z0kv::2], kvs[n_z0kv + 1::2])) if use_cross_kv_cache else None
                    if use_adaln_cache:
                        return self._action_only_forward_cached(z0c, a, ctx, mask, None, t, xkv)
                    return self._action_only_forward_cached(z0c, a, ctx, mask, t, None, xkv)

                name = "step" + ("_adaln" if use_adaln_cache else "") + ("_xkv" if use_cross_kv_cache else "")
                a0 = torch.cat([a_t, a_t], dim=0) if use_cfg else a_t
                graphed = self._graphed(name, step, (a0, act_tables[0] if use_adaln_cache else t0),
                                        (action_context, action_context_mask, *flat))

        for i in range(num_inference_steps):
            if act_tables is not None:
                t_a = act_tables[i]  # the step's AdaLN table replaces t
            else:
                t_a = self.action_scheduler.scale_for_model_input(i, device).expand(n_fwd)
            sigma_a = self.action_scheduler.sigma_for_step(i, device).view(1, 1, 1)
            a_in = torch.cat([a_t, a_t], dim=0) if use_cfg else a_t
            if graphed is not None:
                x_a = graphed(a_in, t_a)
            elif act_tables is not None:
                x_a = self._action_only_forward_cached(z0_cache, a_in, action_context, action_context_mask,
                                                       None, t_a, cross_kv)
            elif z0_cache is not None:
                x_a = self._action_only_forward_cached(z0_cache, a_in, action_context, action_context_mask,
                                                       t_a, None, cross_kv)
            else:
                x_a = self._action_only_forward(z0, a_in, image_context, action_context, context_mask, t_a,
                                                action_context_mask=action_context_mask)
            if use_cfg:
                x_c, x_u = x_a[:bsz], x_a[bsz:]
                x_a = x_u + float(cfg_scale) * (x_c - x_u)
            v_a = self.action_scheduler.x_pred_to_velocity(x_a, a_t, sigma_a)
            a_t = a_t + v_a * dt_a[i]
        return a_t

    def _trainable_state_dict(self):
        return {k: v for k, v in self.state_dict().items() if not k.startswith("backbone.")}

    def save_checkpoint(self, path: str, optimizer=None, step: int | None = None):
        payload = {"model": self._trainable_state_dict(), "metadata": self.metadata, "step": step}
        torch.save(payload, path)

    def load_checkpoint(self, path: str, optimizer=None):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self.load_state_dict(payload["model"] if "model" in payload else payload, strict=False)
