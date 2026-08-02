"""Training-loss difficulty probes.

The scalar training loss says how well the model fits the data on average, which
is not enough to explain a slowdown: flow-matching loss varies by an order of
magnitude across noise levels, and a multi-resolution cache mixes shapes and
durations that are not equally easy. These tallies split the same per-sample
losses the objective already computes along three axes:

- timestep band: which part of the noise schedule is still improving. Low t is
  nearly pure noise (loss stays high and matters little perceptually); high t is
  where fine detail lives.
- bucket: whether some resolution / duration is being left behind.
- clip: which individual training items the model cannot fit.

Everything accumulates in device-side buffers and is only copied to the host on
flush, so the hot loop gains no synchronisation.
"""

import torch


class LossProbe:
    def __init__(self, n_bins, n_buckets, n_clips, device):
        self.n_bins = n_bins
        self._sums, self._counts = {}, {}
        for key, size in (
            ("t", n_bins),
            ("t_x1", n_bins),
            ("bucket", n_buckets),
            ("clip", n_clips),
        ):
            self._sums[key] = torch.zeros(size, device=device)
            self._counts[key] = torch.zeros(size, device=device)

    def _add(self, key, index, values):
        self._sums[key].scatter_add_(0, index, values)
        self._counts[key].scatter_add_(0, index, torch.ones_like(values))

    def update(self, per_sample, t, bucket_ids, clip_ids=None, x1_per_sample=None):
        """Tally one bin. `per_sample` and `t` cover the bin's occupied slots."""
        n = bucket_ids.shape[0]
        loss, t = per_sample[:n].float(), t[:n].float()
        if x1_per_sample is None:
            x1_loss = loss * (1 - t).square()
        else:
            x1_loss = x1_per_sample[:n].float()
        t_band = (t * self.n_bins).long().clamp(max=self.n_bins - 1)
        self._add("t", t_band, loss)
        self._add("t_x1", t_band, x1_loss)
        self._add("bucket", bucket_ids, loss)
        if clip_ids is not None:
            self._add("clip", clip_ids, loss)

    def flush(self):
        """Mean loss per axis since the last flush; unvisited entries are NaN."""
        out = {}
        for key in self._sums:
            counts = self._counts[key]
            out[key] = (self._sums[key] / counts.where(counts > 0, torch.nan)).tolist()
            self._sums[key].zero_()
            counts.zero_()
        return out

    def t_band_label(self, i):
        return f"{i / self.n_bins:.2f}-{(i + 1) / self.n_bins:.2f}"
