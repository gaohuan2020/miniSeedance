"""
Optimizer factory, selected by config train.optimizer:

- "adamw": fused AdamW over all parameters (the original recipe).
- "muon":  Muon (momentum + Newton-Schulz orthogonalization) for the 2D hidden
  weights inside the transformer blocks, aux AdamW for everything else
  (embedders, timestep MLP, output heads, projector, norm gains, biases).
  Follows https://github.com/lavinal712/DiT-Muon (muon.py is Keller Jordan's
  reference implementation; the single-device variant is correct under DDP
  because gradients are already all-reduced, so every rank applies the same
  update).

Every param group carries a "base_lr" key; the trainer rescales group lrs each
step with its warmup/cosine schedule.
"""

import torch

from src.log import get_logger

logger = get_logger(__name__)


def zeropower_via_newtonschulz5(G, steps: int):
    """Newton-Schulz iteration approximating the orthogonalization UV^T of G
    (quintic iteration, runs stably in bfloat16). From Keller Jordan's Muon."""
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)  # spectral norm <= 1
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def muon_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4:  # conv filters: collapse to 2D
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update *= max(1, grad.size(-2) / grad.size(-1)) ** 0.5
    return update


def adam_update(grad, buf1, buf2, step, betas, eps):
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0] ** step)
    buf2c = buf2 / (1 - betas[1] ** step)
    return buf1c / (buf2c.sqrt() + eps)


class MuonWithAuxAdam(torch.optim.Optimizer):
    """Single optimizer covering the whole network: param groups flagged
    use_muon=True get Muon updates, the rest an internal AdamW.
    Single-device variant (each DDP rank applies identical updates).

    Muon params sharing a shape (e.g. the same matrix across transformer
    layers) are stacked and orthogonalized with one batched Newton-Schulz
    run, which is ~5x faster than the per-matrix reference loop."""

    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group.setdefault("lr", 0.02)
                group.setdefault("momentum", 0.95)
                group.setdefault("weight_decay", 0)
            else:
                group.setdefault("lr", 3e-4)
                group.setdefault("betas", (0.9, 0.95))
                group.setdefault("eps", 1e-10)
                group.setdefault("weight_decay", 0)
        super().__init__(param_groups, dict())

    def _shape_buckets(self, params):
        buckets = {}
        for p in params:
            buckets.setdefault(tuple(p.shape), []).append(p)
        return buckets.values()

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            if group["use_muon"]:
                for params in self._shape_buckets(group["params"]):
                    active = [p for p in params if p.grad is not None]
                    if not active:
                        continue
                    grads = torch.stack([p.grad for p in active])
                    if len(active) == len(params):
                        state = self.state[active[0]]
                        if "momentum_stack" not in state:
                            state["momentum_stack"] = torch.zeros_like(grads)
                        update = muon_update(
                            grads, state["momentum_stack"], beta=group["momentum"]
                        )
                    else:
                        # Sparse MoE routing can leave an expert unused. Keep
                        # momentum per active parameter when the bucket is partial.
                        momenta = []
                        for p in active:
                            state = self.state[p]
                            if "sparse_momentum" not in state:
                                state["sparse_momentum"] = torch.zeros_like(p)
                            momenta.append(state["sparse_momentum"])
                        momentum = torch.stack(momenta)
                        update = muon_update(grads, momentum, beta=group["momentum"])
                        for p, value in zip(active, momentum.unbind(0)):
                            self.state[p]["sparse_momentum"].copy_(value)
                    if group["weight_decay"]:
                        torch._foreach_mul_(active, 1 - group["lr"] * group["weight_decay"])
                    torch._foreach_add_(active, list(update.to(active[0].dtype).unbind(0)),
                                        alpha=-group["lr"])
            else:
                params = [p for p in group["params"] if p.grad is not None]
                for p in params:
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                         state["step"], group["betas"], group["eps"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])


def build_optimizer(model, tcfg):
    """Build the optimizer described by tcfg["optimizer"]; every group gets a
    "base_lr" so the trainer's schedule can rescale all groups uniformly."""
    kind = tcfg["optimizer"]
    if kind == "adamw":
        try:
            opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=0.0, fused=True)
        except RuntimeError:
            opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=0.0)
    elif kind == "muon":
        # Muon only fits hidden 2D weights (transformer blocks); embedders,
        # heads, gains and biases keep AdamW, as in DiT-Muon. mHC's tiny n x n
        # static tables (b_res / modality_scale_shift) are bias-like despite
        # being 2D.
        def is_muon(n, p):
            return (
                p.ndim >= 2
                and n.startswith(("blocks.", "context_refiner."))
                and ".mlp.router." not in n
                and not n.endswith("b_res")
                and not n.endswith("modality_scale_shift")
            )

        muon_params = [p for n, p in model.named_parameters() if is_muon(n, p) and p.requires_grad]
        adamw_params = [p for n, p in model.named_parameters()
                        if not is_muon(n, p) and p.requires_grad]
        assert muon_params, "no block weights found for Muon"
        opt = MuonWithAuxAdam([
            {"params": muon_params, "use_muon": True,
             "lr": tcfg["muon_lr"], "momentum": tcfg["muon_momentum"], "weight_decay": 0},
            {"params": adamw_params, "use_muon": False,
             "lr": tcfg["lr"], "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0},
        ])
        logger.info("Muon: %d hidden matrices (%.1fM params) | aux AdamW: %d tensors (%.1fM params)",
                    len(muon_params), sum(p.numel() for p in muon_params) / 1e6,
                    len(adamw_params), sum(p.numel() for p in adamw_params) / 1e6)
    else:
        raise ValueError(f"unknown optimizer {kind!r} (adamw | muon)")

    for group in opt.param_groups:
        group["base_lr"] = group["lr"]
    return opt
