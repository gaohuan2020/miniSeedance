"""
Frozen text encoders for the minimal text-to-video model. Per-token features
condition the DiT via the joint text-video sequence.

- CLIPTextEncoder: CLIP (openai/clip-vit-base-patch32), 512-d tokens.
- T5GemmaTextEncoder: T5Gemma encoder (ported from daVinci-MagiHuman
  inference/model/t5_gemma/), loaded from a local checkpoint directory.
"""

import contextlib

import torch
from transformers import AutoTokenizer, CLIPTextModel, CLIPTokenizer
from transformers.utils import logging as hf_logging


@contextlib.contextmanager
def _quiet_hf_logging():
    """Silence transformers' load report: loading CLIPTextModel from a full
    CLIP checkpoint intentionally skips the vision-tower weights, which would
    otherwise be printed as UNEXPECTED keys."""
    verbosity = hf_logging.get_verbosity()
    hf_logging.set_verbosity_error()
    try:
        yield
    finally:
        hf_logging.set_verbosity(verbosity)


class CLIPTextEncoder:
    """Frozen CLIP text encoder."""

    def __init__(self, device, name="openai/clip-vit-base-patch32", dtype=torch.float32):
        self.tokenizer = CLIPTokenizer.from_pretrained(name)
        with _quiet_hf_logging():
            self.model = CLIPTextModel.from_pretrained(name, torch_dtype=dtype)
        self.model = self.model.to(device=device)
        self.model.eval()
        self.model.requires_grad_(False)
        self.device = device
        self.embed_dim = self.model.config.hidden_size

    @torch.no_grad()
    def encode_tokens(self, texts: list[str], max_length: int = 16) -> torch.Tensor:
        """Per-token features (B, max_length, hidden) for joint-sequence models."""
        tokens = self.tokenizer(
            texts,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        out = self.model(**tokens)
        return out.last_hidden_state.float()

    @torch.no_grad()
    def encode_pooled(self, texts: list[str], max_length: int = 16) -> torch.Tensor:
        """Pooled CLIP text features for semantic retrieval."""
        tokens = self.tokenizer(
            texts,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        return self.model(**tokens).pooler_output.float()


class T5GemmaTextEncoder:
    """Frozen T5Gemma encoder (MagiHuman's text encoder, minus CPU offload).

    MagiHuman uses t5gemma-9b-9b-ul2; any T5Gemma checkpoint directory works
    (e.g. checkpoints/t5gemma-2b-2b-ul2-it). Only the encoder stack is loaded.
    """

    def __init__(self, device, model_path="checkpoints/t5gemma-2b-2b-ul2-it", dtype=torch.bfloat16):
        from transformers.models.t5gemma import T5GemmaEncoderModel

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        with _quiet_hf_logging():
            self.model = T5GemmaEncoderModel.from_pretrained(
                model_path, is_encoder_decoder=False, dtype=dtype
            ).to(device)
        self.model.eval()
        self.model.requires_grad_(False)
        self.device = device
        self.embed_dim = self.model.config.hidden_size

    @torch.no_grad()
    def encode_tokens(self, texts: list[str], max_length: int = 16) -> torch.Tensor:
        """Per-token features (B, max_length, hidden) for joint-sequence models."""
        tokens = self.tokenizer(
            texts,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        out = self.model(**tokens)
        return out.last_hidden_state.float()
