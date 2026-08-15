"""ELF-style continuous flow for draft-block prediction (arXiv 2605.10938),
adapted to speculative decoding with KV-injected target conditioning.

Conventions (ELF): t=0 noise, t=1 data.  z_t = t * x + (1 - t) * eps.
x = (E[y] - mu) / sigma, scalar stats over the target embedding matrix.
eps ~ N(0, noise_scale^2 I); ELF uses noise_scale=2 for the denoise branch.

Training, per block (anchor slot always clean, never corrupted):
  denoise branch (prob 0.8):
      t ~ logit-normal(mean=-1.5, std=0.8) per block
      loss = || (x_hat - z_t)/(1 - t) - (x - eps) ||^2      (velocity MSE,
      x-prediction parameterization; no CE supervision on this branch)
  decode branch (prob 0.2):
      t = 1, mode = decode; per-TOKEN cleanliness p ~ logit-normal(0.8, 0.8)
      input z~ = p * x + (1 - p) * eps_dec (decode_noise_scale=1)
      loss = CE(frozen lm_head(trunk output), y)            (no MSE)

Both branches use position-decay weights w_o = exp(-(o-1)/gamma) (spec-decoding
asymmetry: early block positions gate acceptance).

Inference: K total draft forwards = (K-1) denoise ODE/SDE steps + 1 decode
forward at t=1. K=1 = decode straight from noise (mask-baseline equivalence
control). Self-conditioning chains x_hat between denoise steps.
"""
import math
from dataclasses import dataclass

import torch


@dataclass
class FlowConfig:
    denoise_prob: float = 0.8
    t_mean: float = -1.5          # logit-normal params for denoise-branch t
    t_std: float = 0.8
    p_mean: float = 0.8           # logit-normal params for decode-branch per-token p
    p_std: float = 0.8
    noise_scale: float = 2.0      # denoise-branch (and inference init) noise std
    decode_noise_scale: float = 1.0
    self_cond: bool = True        # 50% self-conditioning on denoise branch
    sde_gamma: float = 0.0        # >0 enables SDE-style noise re-injection at inference
    grid: str = "logit-normal"    # inference time grid: logit-normal quantiles | uniform

MODE_DENOISE = 0
MODE_DECODE = 1


def emb_stats(embed_weight: torch.Tensor) -> tuple[float, float]:
    w = embed_weight.float()
    return w.mean().item(), w.std().item()


def sample_logit_normal(shape, mean, std, device, generator=None):
    return torch.sigmoid(torch.randn(shape, device=device, generator=generator) * std + mean)


def norm_inv_cdf(p: float) -> float:
    """Inverse standard normal CDF (Acklam-lite via erfinv)."""
    return math.sqrt(2.0) * torch.erfinv(torch.tensor(2.0 * p - 1.0)).item()


def inference_t_grid(total_steps: int, cfg: FlowConfig) -> list[float]:
    """Time points: denoise forwards run at grid[0..K-2]; final state reaches
    t=1 and the decode forward consumes it. total_steps includes the decode
    forward. Returns grid of length total_steps (last entry is 1.0)."""
    k = total_steps - 1  # number of denoise steps
    if k <= 0:
        return [1.0]
    if k == 1:
        return [0.0, 1.0]
    if cfg.grid == "uniform":
        inner = [i / k for i in range(1, k)]
    else:  # logit-normal quantiles of the training-time t distribution
        inner = [
            1.0 / (1.0 + math.exp(-(cfg.t_mean + cfg.t_std * norm_inv_cdf(i / k))))
            for i in range(1, k)
        ]
        inner = sorted(inner)
    return [0.0] + inner + [1.0]


def velocity(x_hat: torch.Tensor, z: torch.Tensor, t: float) -> torch.Tensor:
    return (x_hat - z) / max(1.0 - t, 1e-3)


# ---------------- RCF (rollout-consistent continuous flow) ----------------
# Endpoint field derived from the frozen head's posterior: x1_hat = E_p[e_y].
# Single source of truth for trainer AND generator (U2 identity).

RCF_T_GRIDS = {
    1: [0.0],
    2: [0.0, 0.5],
    3: [0.0, 0.45, 0.75],
    4: [0.0, 0.40, 0.65, 0.85],
    8: [0.0, 0.25, 0.45, 0.60, 0.72, 0.82, 0.90, 0.95],
}


def endpoint_from_logits(logits: torch.Tensor, embed_weight: torch.Tensor,
                         T_r: float = 1.0, top_m: int = 64,
                         chunk: int = 8192) -> torch.Tensor:
    """logits: [..., V] -> expected embedding under top-m renormalized
    posterior: [..., D] (raw embedding space, fp32)."""
    shape = logits.shape[:-1]
    flat = logits.reshape(-1, logits.shape[-1]).float()
    outs = []
    for i in range(0, flat.shape[0], chunk):
        sl = flat[i:i + chunk]
        p = torch.softmax(sl / T_r, dim=-1)
        v, idx = p.topk(top_m, dim=-1)
        v = v / v.sum(-1, keepdim=True)
        e = embed_weight[idx].float()                     # [n, m, D]
        outs.append(torch.bmm(v.unsqueeze(1), e).squeeze(1))
    return torch.cat(outs, 0).view(*shape, -1)


def integrator_step(x: torch.Tensor, x1_hat: torch.Tensor,
                    t: float, t_next: float) -> torch.Tensor:
    """Exact exponential-integrator/DDIM update for endpoint parameterization
    on the rectified path. From x=0 at t=0: x(t1) = t1 * x1_hat."""
    return x + (t_next - t) / max(1.0 - t, 1e-3) * (x1_hat - x)
