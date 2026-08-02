"""Lossless STFT audio tokenizer (audio_vae.type = "stft").

Unlike a pretrained VAE (e.g. Stable Audio's Oobleck, 2048x waveform
compression with a real reconstruction ceiling), this codec is exactly
invertible: encode is a complex STFT with amplitude compression, packed as
(re, im) channels; decode is the inverse compression + iSTFT (Hann window
with 2x overlap satisfies COLA, i.e. perfect reconstruction). The diffusion
model then carries the full audio fidelity burden -- more tokens and wider
channels, but no frozen-decoder quality bound.

Token geometry for 44.1 kHz stereo, n_fft=1024, hop=512:
    (B, 2, N) -> (B, 4 * (n_fft/2 + 1), N/hop + 1)  = (B, 2052, ~86/s)
Amplitude compression |X|^0.3 tames the spectral dynamic range so the
per-channel dataset normalization applied by latent caching stays sane.
"""

import torch


class STFTAudioCodec:
    def __init__(self, sample_rate=44100, n_fft=1024, hop=512, comp=0.3, device="cuda"):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop = hop
        self.comp = comp
        self.device = device
        self.window = torch.hann_window(n_fft, device=device)
        self.n_bins = n_fft // 2 + 1
        self.latent_dim = 2 * 2 * self.n_bins  # stereo x (re, im) x bins

    @torch.no_grad()
    def encode(self, waveform):
        """(B, 2, N) in [-1, 1] -> (B, 4 * bins, L); N is padded to a hop multiple."""
        b, ch, n = waveform.shape
        pad = (-n) % self.hop
        if pad:
            waveform = torch.nn.functional.pad(waveform, (0, pad))
        x = waveform.to(self.device).reshape(b * ch, -1)
        spec = torch.stft(x.float(), self.n_fft, self.hop, window=self.window,
                          center=True, return_complex=True)          # (B*ch, F, L)
        spec = spec * spec.abs().clamp_min(1e-8).pow(self.comp - 1)  # |X|^comp phase-preserving
        z = torch.view_as_real(spec)                                 # (B*ch, F, L, 2)
        return z.permute(0, 3, 1, 2).reshape(b, 2 * ch * self.n_bins, -1)

    @torch.no_grad()
    def decode(self, latents):
        """(B, 4 * bins, L) -> (B, 2, (L - 1) * hop) in [-1, 1]."""
        b, c, l = latents.shape
        ch = c // (2 * self.n_bins)
        z = latents.to(self.device).float().reshape(b * ch, 2, self.n_bins, l).permute(0, 2, 3, 1)
        spec = torch.view_as_complex(z.contiguous())
        spec = spec * spec.abs().clamp_min(1e-8).pow(1.0 / self.comp - 1)
        wave = torch.istft(spec, self.n_fft, self.hop, window=self.window,
                           center=True, length=(l - 1) * self.hop)
        return wave.reshape(b, ch, -1).clamp(-1, 1)
