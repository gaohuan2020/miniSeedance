"""
Training-time video generation metrics.

- FrameFID: frame-level FID with InceptionV3 pool features. The primary
  metric is the per-video FID mean: each generated video's frames form their
  own Gaussian, compared against the shared real-frame Gaussian (low-rank
  trick: eig(S1 @ S2) via the T x T matrix C^T S2 C / (T-1), no 2048^2 sqrtm
  per video). A pooled all-frames FID is logged as a secondary value.
- VideoFVD: FVD-style Frechet distance on video-level features from a
  Kinetics-400 pretrained r3d_18 (512-d per video). Not the official
  I3D-logits FVD, but the same construction.
- CLIPScoreMetric: mean CLIP cosine similarity (x100) between evenly sampled
  frames and the video's caption (ViT-B/32).
- AudioFAD: Frechet Audio Distance on VGGish embeddings (AudioSet-pretrained,
  128-d per 0.96 s frame), the standard FAD construction.

Real and generated videos go through the same VAE decode so the Frechet
metrics ignore the VAE gap. With few videos per side these are noisy trend
metrics, not paper-grade numbers.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def frechet_distance(mu1, sigma1, mu2, sigma2):
    import warnings

    from scipy import linalg

    diff = mu1 - mu2
    if not sigma1.any() or not sigma2.any():  # single-sample side: sqrt(S1 S2) = 0
        return float(diff @ diff + np.trace(sigma1 + sigma2))
    with warnings.catch_warnings():
        # rank-deficient covariances (few samples) are expected here
        warnings.simplefilter("ignore", linalg.LinAlgWarning)
        covmean = linalg.sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))


def _stats(feats):
    """(N, D) features -> (mu, sigma); zero covariance when N == 1."""
    mu = feats.mean(axis=0)
    if feats.shape[0] < 2:
        sigma = np.zeros((feats.shape[1], feats.shape[1]), dtype=np.float64)
    else:
        sigma = np.cov(feats, rowvar=False)
    return mu, sigma


class FrameFID:
    def __init__(self, device):
        from torchvision.models import Inception_V3_Weights, inception_v3

        model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        model.fc = nn.Identity()
        self.model = model.eval().to(device).requires_grad_(False)
        self.device = device
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def features(self, frames, batch_size=128):
        """frames: uint8 (N, H, W, 3) -> (N, 2048) numpy."""
        feats = []
        for i in range(0, len(frames), batch_size):
            x = torch.from_numpy(frames[i : i + batch_size]).to(self.device)
            x = x.permute(0, 3, 1, 2).float() / 255.0
            x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
            x = (x - self.mean) / self.std
            feats.append(self.model(x).cpu().numpy())
        return np.concatenate(feats)

    @staticmethod
    def stats(feats):
        """(N, D) features -> (mu (D,), sigma (D, D))."""
        return _stats(feats)

    @staticmethod
    def fid(feats_real, feats_fake):
        """Pooled FID over all frames of each side."""
        return frechet_distance(*_stats(feats_real), *_stats(feats_fake))

    @staticmethod
    def video_fid(mu_real, sigma_real, video_feats):
        """FID between one video's frames (T, D) and the real Gaussian.

        Uses the low-rank identity eig(S1 @ S2) = eig(C^T S2 C / (T-1)) with
        S1 = C C^T / (T-1), C the centered per-video features, so only a
        T x T eigendecomposition is needed."""
        t = video_feats.shape[0]
        mu1 = video_feats.mean(axis=0)
        c = (video_feats - mu1).T.astype(np.float64)  # (D, T)
        diff = mu1 - mu_real

        tr_s1 = float((c ** 2).sum()) / (t - 1)
        tr_s2 = float(np.trace(sigma_real))
        small = c.T @ sigma_real @ c / (t - 1)  # (T, T), eigs of S1 @ S2
        eigs = np.linalg.eigvals(small).real.clip(min=0)
        tr_sqrt = float(np.sqrt(eigs).sum())
        return float(diff @ diff + tr_s1 + tr_s2 - 2 * tr_sqrt)

    @staticmethod
    def mean_video_fid(feats_real, video_feats_list):
        """Mean of per-video FIDs against the shared real-frame Gaussian."""
        mu_real, sigma_real = FrameFID.stats(feats_real)
        sigma_real = sigma_real.astype(np.float64)
        fids = [FrameFID.video_fid(mu_real, sigma_real, vf) for vf in video_feats_list]
        return float(np.mean(fids))


class VideoFVD:
    """FVD-style Frechet distance on Kinetics-400 r3d_18 video features."""

    MEAN = (0.43216, 0.394666, 0.37645)
    STD = (0.22803, 0.22145, 0.216989)

    def __init__(self, device):
        from torchvision.models.video import R3D_18_Weights, r3d_18

        model = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
        model.fc = nn.Identity()
        self.model = model.eval().to(device).requires_grad_(False)
        self.device = device
        self.mean = torch.tensor(self.MEAN, device=device).view(1, 3, 1, 1, 1)
        self.std = torch.tensor(self.STD, device=device).view(1, 3, 1, 1, 1)

    @torch.no_grad()
    def features(self, videos, batch_size=8):
        """videos: uint8 (B, T, H, W, 3) -> (B, 512) numpy."""
        feats = []
        for i in range(0, len(videos), batch_size):
            x = torch.from_numpy(videos[i : i + batch_size]).to(self.device)
            x = x.permute(0, 4, 1, 2, 3).float() / 255.0  # (b, 3, t, h, w)
            b, c, t = x.shape[:3]
            frames = F.interpolate(
                x.transpose(1, 2).reshape(b * t, c, *x.shape[-2:]),
                size=(112, 112), mode="bilinear", align_corners=False,
            )
            x = frames.reshape(b, t, c, 112, 112).transpose(1, 2)
            x = (x - self.mean) / self.std
            feats.append(self.model(x).cpu().numpy())
        return np.concatenate(feats)

    @staticmethod
    def fvd(feats_real, feats_fake):
        return frechet_distance(*_stats(feats_real), *_stats(feats_fake))


class CLIPScoreMetric:
    """Mean CLIP cosine similarity (x100) between video frames and caption."""

    MEAN = (0.48145466, 0.4578275, 0.40821073)
    STD = (0.26862954, 0.26130258, 0.27577711)

    def __init__(self, device, name="openai/clip-vit-base-patch32"):
        from transformers import CLIPModel, CLIPTokenizer

        self.model = CLIPModel.from_pretrained(name).eval().to(device).requires_grad_(False)
        self.tokenizer = CLIPTokenizer.from_pretrained(name)
        self.device = device
        self.mean = torch.tensor(self.MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(self.STD, device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def score(self, videos, captions, frames_per_video=8):
        """videos: uint8 (B, T, H, W, 3); captions: list of B strings -> float."""
        b, t = videos.shape[:2]
        idx = np.linspace(0, t - 1, min(frames_per_video, t)).round().astype(int)
        frames = torch.from_numpy(videos[:, idx]).to(self.device)  # (b, f, h, w, 3)
        f = frames.shape[1]
        x = frames.reshape(b * f, *frames.shape[2:]).permute(0, 3, 1, 2).float() / 255.0
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        img = self.model.get_image_features(pixel_values=x)
        if not torch.is_tensor(img):  # transformers >= 5 returns a ModelOutput
            img = img.pooler_output
        img = F.normalize(img, dim=-1).reshape(b, f, -1)

        tok = self.tokenizer(captions, padding=True, truncation=True, return_tensors="pt").to(self.device)
        txt = self.model.get_text_features(**tok)
        if not torch.is_tensor(txt):
            txt = txt.pooler_output
        txt = F.normalize(txt, dim=-1)  # (b, d)

        sim = (img * txt.unsqueeze(1)).sum(-1)  # (b, f)
        return float(sim.mean().item() * 100)


class AudioFAD:
    """Frechet Audio Distance on VGGish embeddings (torch.hub
    harritaylor/torchvggish; raw 128-d embeddings, no PCA postprocessing)."""

    def __init__(self, device):
        self.model = torch.hub.load("harritaylor/torchvggish", "vggish", trust_repo=True)
        self.model.postprocess = False
        self.model.eval().to(device).requires_grad_(False)
        self.device = device

    @torch.no_grad()
    def features(self, waves, sample_rate):
        """waves: float (B, C, N) array in [-1, 1] -> (B * frames, 128) numpy.
        Stereo is averaged to mono; VGGish resamples to 16 kHz internally."""
        feats = []
        for wav in waves:
            mono = np.asarray(wav, dtype=np.float32).mean(axis=0)
            emb = self.model(mono, fs=sample_rate)  # (frames, 128)
            feats.append(emb.cpu().numpy())
        return np.concatenate(feats)

    @staticmethod
    def fad(feats_real, feats_fake):
        return frechet_distance(*_stats(feats_real), *_stats(feats_fake))
