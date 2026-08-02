"""
Training objectives, selected by config train.objective.

Both share the path/sign conventions of Self-Flow:
    x_t = t * x1 + (1 - t) * x0, v = x1 - x0
    pred = model(x_t, timesteps=1 - t, vector);  loss = MSE(-pred, v)

- FlowMatching: plain conditional flow matching; per-modality losses on the
  packed [video | audio] sequence (padding channels excluded).
- SelfFlow: flow matching under Dual-Timestep Scheduling (two timesteps from
  p(t); a random token subset takes the second one) + self-distillation of the
  student's features against an EMA teacher that sees the cleaner timestep.

Under sequence packing the noised tensor is a fixed-budget bin whose unused
tail is marked `modality == PAD`. Those slots are excluded from both the FM
and distill losses (and sit in their own FA4 segment, so they also cannot
attend to real tokens). Every loss here averages *per sample first*, so a bin
of one long clip and a bin of four short ones report the same scale.

An objective is a callable:
  (ddp_model, ema, x1_tok, y, plan, step) -> (loss, metrics dict, probe dict)
where `plan` is a PackPlan. The probe carries the still-unreduced per-sample
video loss and its timestep so callers can attribute difficulty to individual
clips and noise levels; it is detached and never affects training.
"""

import torch
import torch.nn.functional as F

from src.utils import (
    AUDIO,
    PAD,
    VIDEO,
    PackPlan,
    masked_per_sample_losses,
    masked_per_sample_mean,
    timeshift,
)


class FlowMatching:
    needs_unused_param_handling = True  # never touches the distillation projector

    def __init__(self, tcfg: dict, depth: int, video_channels: int, audio_channels: int = 0):
        self.video_weight = tcfg["video_weight"]
        self.audio_weight = tcfg["audio_weight"]
        self.t_sampler = tcfg["t_sampler"]
        self.t_logit_mean = tcfg["t_logit_mean"]
        self.t_logit_std = tcfg["t_logit_std"]
        self.t_shift = tcfg["t_shift"]
        self.t_low_snr = tcfg["t_low_snr"]
        self.t_low_snr_width = tcfg["t_low_snr_width"]
        self.t_clean_mid = tcfg.get("t_clean_mid", 0.0)
        self.t_clean_mid_range = tcfg.get("t_clean_mid_range", [0.75, 0.90])
        self.t_clean_high = tcfg.get("t_clean_high", 0.0)
        self.t_clean_high_range = tcfg.get("t_clean_high_range", [0.90, 0.98])
        clean_total = self.t_clean_mid + self.t_clean_high
        if not 0 <= clean_total <= 1:
            raise ValueError("t_clean_mid + t_clean_high must be in [0, 1]")
        for name, bounds in (
            ("t_clean_mid_range", self.t_clean_mid_range),
            ("t_clean_high_range", self.t_clean_high_range),
        ):
            if len(bounds) != 2 or not 0 <= bounds[0] < bounds[1] <= 1:
                raise ValueError(f"{name} must be [lo, hi] within [0, 1]")
        self.video_channels = video_channels
        self.audio_channels = audio_channels

    def describe(self):
        return (f"flow matching (t_sampler={self.describe_t()}, "
                f"video_weight={self.video_weight}, audio_weight={self.audio_weight})")

    def describe_t(self):
        base = (self.t_sampler if self.t_sampler != "lognorm"
                else f"lognorm(mean={self.t_logit_mean}, std={self.t_logit_std})")
        if self.t_shift != 1.0:
            base += f" shift={self.t_shift}"
        if self.t_low_snr:
            base += f" low_snr={self.t_low_snr}@[0,{self.t_low_snr_width}]"
        if self.t_clean_mid or self.t_clean_high:
            base += (
                f" clean_mix={self.t_clean_mid}@{self.t_clean_mid_range}"
                f"+{self.t_clean_high}@{self.t_clean_high_range}"
            )
        return base

    def sample_t(self, n, device):
        if self.t_sampler == "lognorm":
            # t=1 is clean data, so a negative logit mean shifts the budget
            # toward the noisier end, where the loss is still moving.
            z = torch.randn(n, device=device) * self.t_logit_std + self.t_logit_mean
            t = torch.sigmoid(z)
        else:
            t = torch.rand(n, device=device)
        # trainshift; for the lognorm sampler this is exactly a `log a` offset on
        # the logit mean, since logit(s(a, u)) = log a + logit(u).
        t = timeshift(t, self.t_shift)
        if self.t_low_snr:
            # A logit-normal's density vanishes at both extremes however it is
            # shifted, so the noisiest timesteps -- where sampling starts -- stay
            # barely trained. Redraw a fraction of them uniformly over the
            # noisiest interval, which is [0, width] here since t=0 is noise.
            redraw = torch.rand(n, device=device) < self.t_low_snr
            low = torch.rand(n, device=device) * self.t_low_snr_width
            t = torch.where(redraw, low, t)
        # Optionally reserve explicit budget for the clean end. This replacement
        # is deliberately unweighted: the goal is to change the optimization
        # target toward fine-detail timesteps, not estimate the old objective.
        clean_total = self.t_clean_mid + self.t_clean_high
        if clean_total:
            choice = torch.rand(n, device=device)
            mid_mask = choice < self.t_clean_mid
            high_mask = (choice >= self.t_clean_mid) & (choice < clean_total)
            mid_lo, mid_hi = self.t_clean_mid_range
            high_lo, high_hi = self.t_clean_high_range
            mid = torch.rand(n, device=device) * (mid_hi - mid_lo) + mid_lo
            high = torch.rand(n, device=device) * (high_hi - high_lo) + high_lo
            t = torch.where(mid_mask, mid, t)
            t = torch.where(high_mask, high, t)
        return t

    def fm_losses(self, pred, target, t_tok, plan: PackPlan):
        """Per-modality FM losses; channel padding and bin PAD slots are excluded.

        Averaged per sample and then over samples, so the loss does not tilt
        toward whichever clip contributed the most tokens to the bin. Also
        returns the per-sample video losses for the difficulty probe.
        """
        err = (-pred.float() - target) ** 2
        args = (plan.segment, plan.max_seqs)
        per_v, valid = masked_per_sample_losses(err, plan.modality == VIDEO, self.video_channels, *args)
        per_v_x1, _ = masked_per_sample_losses(
            err * (1 - t_tok).square().unsqueeze(-1),
            plan.modality == VIDEO,
            self.video_channels,
            *args,
        )
        v = valid.to(err.dtype)
        loss_v = (per_v * v).sum() / v.sum().clamp(min=1.0)
        if self.audio_channels <= 0:
            return (
                loss_v,
                torch.zeros((), device=pred.device),
                per_v[0].detach(),
                per_v_x1[0].detach(),
            )
        loss_a = masked_per_sample_mean(err, plan.modality == AUDIO, self.audio_channels, *args)
        return loss_v, loss_a, per_v[0].detach(), per_v_x1[0].detach()

    def __call__(self, ddp_model, ema, x1_tok, y, plan: PackPlan, step):
        x0_tok = torch.randn_like(x1_tok)
        v_tok = x1_tok - x0_tok
        # One t per bin slot rather than per occupied slot: a data-dependent
        # count would sync the device and re-specialize the compiled graph.
        t = self.sample_t(plan.max_seqs, x1_tok.device)

        # Per-token t via the sample index; PAD slots get t=0 (pure noise) so
        # their targets stay well-defined even though the loss ignores them.
        t_tok = t[plan.segment].unsqueeze(0).expand(x1_tok.shape[0], -1)
        t_tok = t_tok.masked_fill(plan.modality.unsqueeze(0) == PAD, 0.0)
        t_ = t_tok.unsqueeze(-1)
        xt_tok = t_ * x1_tok + (1 - t_) * x0_tok
        pred = ddp_model(xt_tok, timesteps=1 - t_tok, vector=y, pack_plan=plan)
        loss_v, loss_a, per_sample_v, per_sample_v_x1 = self.fm_losses(
            pred, v_tok, t_tok, plan
        )
        loss = self.video_weight * loss_v + self.audio_weight * loss_a
        metrics = {"fm_video": loss_v.detach(), "fm_audio": loss_a.detach()}
        return loss, metrics, {
            "video_per_sample": per_sample_v,
            "video_x1_per_sample": per_sample_v_x1,
            "t": t.detach(),
        }


class SelfFlow(FlowMatching):
    needs_unused_param_handling = False

    def __init__(self, tcfg: dict, depth: int, video_channels: int, audio_channels: int = 0):
        super().__init__(tcfg, depth, video_channels, audio_channels)
        self.mask_ratio = tcfg["mask_ratio"]
        self.distill_weight = tcfg["distill_weight"]
        self.distill_warmup = tcfg["distill_warmup"]
        # Self-Flow's layer ratios: student at 0.3 * depth, teacher at 0.7
        self.student_layer = tcfg["student_layer"] or max(1, round(depth * 0.3))
        self.teacher_layer = tcfg["teacher_layer"] or max(1, round(depth * 0.7))

    def describe(self):
        return (f"Self-Flow (mask_ratio={self.mask_ratio}, distill_weight={self.distill_weight}, "
                f"distill_warmup={self.distill_warmup}, "
                f"student_layer={self.student_layer}, teacher_layer={self.teacher_layer}, "
                f"video_weight={self.video_weight}, audio_weight={self.audio_weight}, "
                f"t_sampler={self.describe_t()})")

    def __call__(self, ddp_model, ema, x1_tok, y, plan: PackPlan, step):
        b, x_len, _ = x1_tok.shape
        device = x1_tok.device
        x0_tok = torch.randn_like(x1_tok)
        v_tok = x1_tok - x0_tok

        # Dual-Timestep Scheduling: two timesteps drawn from the same p(t), so
        # every token's marginal timestep distribution is still p(t). Masked
        # tokens take s, the rest take t. Pinning masked tokens to pure noise
        # instead is the "naive masking" variant the paper measures as harmful.
        t = self.sample_t(plan.max_seqs, device)
        s = self.sample_t(plan.max_seqs, device)

        # Per-token timesteps; bin PAD slots get t=0 (their targets stay
        # well-defined even though the loss ignores them).
        pad = plan.modality.unsqueeze(0) == PAD
        t_tok = t[plan.segment].unsqueeze(0).expand(b, -1)
        s_tok = s[plan.segment].unsqueeze(0).expand(b, -1)
        drop = torch.rand(b, x_len, device=device) < self.mask_ratio
        t_tok = torch.where(drop, s_tok, t_tok).masked_fill(pad, 0.0)
        t_ = t_tok.unsqueeze(-1)
        xt_tok = t_ * x1_tok + (1 - t_) * x0_tok

        pred, z_student = ddp_model(
            xt_tok, timesteps=1 - t_tok, vector=y, pack_plan=plan,
            return_features=self.student_layer,
        )
        loss_v, loss_a, per_sample_v, per_sample_v_x1 = self.fm_losses(
            pred, v_tok, t_tok, plan
        )
        loss_fm = self.video_weight * loss_v + self.audio_weight * loss_a

        # The EMA teacher sees the cleaner of the two levels -- t=1 is clean data
        # here, so that is max(t, s) -- applied uniformly to every token. This is
        # the information asymmetry the student has to close. features_only stops
        # the teacher at teacher_layer (skips later blocks + heads).
        with torch.no_grad():
            t_clean = torch.maximum(t, s)
            t_full = t_clean[plan.segment].unsqueeze(0).expand(b, -1)
            t_full = t_full.masked_fill(pad, 0.0)
            xt_full_tok = t_full.unsqueeze(-1) * x1_tok + (1 - t_full.unsqueeze(-1)) * x0_tok
            _, z_teacher = ema(
                xt_full_tok, timesteps=1 - t_full, vector=y, pack_plan=plan,
                return_raw_features=self.teacher_layer, features_only=True,
            )
        # Cosine distance only over real (video|audio) tokens, per sample.
        cos = F.cosine_similarity(z_student.float(), z_teacher.float(), dim=-1)
        loss_distill = masked_per_sample_mean(
            (1 - cos).unsqueeze(-1), plan.modality != PAD, 1, plan.segment, plan.max_seqs
        )

        w = self.distill_weight
        if self.distill_warmup:
            w *= min(1.0, step / self.distill_warmup)
        loss = loss_fm + w * loss_distill
        metrics = {
            "selfflow_video": loss_v.detach(), "selfflow_audio": loss_a.detach(),
            "distill": loss_distill.detach(),
        }
        if self.distill_warmup:
            # While the distill term is ramping, a flat total loss can hide a
            # falling FM loss, so split the two apart. At a constant weight both
            # are exact functions of what is already logged.
            metrics["distill_weighted"] = w * loss_distill.detach()
            metrics["loss_fm"] = loss_fm.detach()
        return loss, metrics, {
            "video_per_sample": per_sample_v,
            "video_x1_per_sample": per_sample_v_x1,
            "t": t.detach(),
        }


OBJECTIVES = {"fm": FlowMatching, "selfflow": SelfFlow}
