"""
Component registries: every pluggable piece (DiT, video VAE, audio VAE, text
encoder) is selected by the `type` field of its config section and built by a
factory registered here. Adding a new implementation = write the class and
register a factory; no call-site changes.

Factory signature: factory(section_cfg: dict, device: str) -> component.
DiT factories additionally receive runtime channel dims as kwargs.
"""


def _make_registry(kind):
    registry = {}

    def register(name):
        def wrap(factory):
            registry[name] = factory
            return factory
        return wrap

    def build(section, device, **kwargs):
        type_name = section["type"]
        if type_name not in registry:
            raise ValueError(f"Unknown {kind} type {type_name!r}; available: {sorted(registry)}")
        return registry[type_name](section, device, **kwargs)

    return registry, register, build


TEXT_ENCODERS, register_text_encoder, _build_text_encoder = _make_registry("text encoder")
VIDEO_VAES, register_video_vae, _build_video_vae = _make_registry("video VAE")
AUDIO_VAES, register_audio_vae, _build_audio_vae = _make_registry("audio VAE")
DIT_MODELS, register_dit, _build_dit = _make_registry("DiT")


# ---- built-in factories (lazy imports keep heavy deps optional) ----

@register_text_encoder("clip")
def _clip(cfg, device):
    from src.models.text_encoders import CLIPTextEncoder

    return CLIPTextEncoder(device, name=cfg["name_or_path"])


@register_text_encoder("t5gemma")
def _t5gemma(cfg, device):
    from src.models.text_encoders import T5GemmaTextEncoder

    return T5GemmaTextEncoder(device, model_path=cfg["name_or_path"])


@register_video_vae("wan2.2")
def _wan22(cfg, device):
    from src.models.video_vae import Wan2_2_VAE

    return Wan2_2_VAE(vae_pth=cfg["path"], device=device)


@register_audio_vae("stable_audio")
def _stable_audio(cfg, device):
    from src.models.audio_vae import StableAudioVAE

    return StableAudioVAE(model_path=cfg["path"], device=device)


@register_audio_vae("stft")
def _stft_codec(cfg, device):
    from src.models.audio_stft import STFTAudioCodec

    return STFTAudioCodec(
        sample_rate=cfg.get("sample_rate", 44100),
        n_fft=cfg.get("n_fft", 1024), hop=cfg.get("hop", 512),
        comp=cfg.get("comp", 0.3), device=device,
    )


@register_dit("video_dit")
def _video_dit(cfg, device, *, video_in_channels, text_in_channels, audio_in_channels=None):
    from src.models.dit import TextToVideoDiT

    return TextToVideoDiT(
        video_in_channels=video_in_channels,
        text_in_channels=text_in_channels,
        audio_in_channels=audio_in_channels,
        hidden_size=cfg["hidden_size"],
        depth=cfg["depth"],
        num_heads=cfg["num_heads"],
        attention=cfg.get("attention", "flash"),
        adaln=cfg.get("adaln", "per_block"),
        mhc=cfg.get("mhc", 0),
        pooled_text=cfg.get("pooled_text", False),
        attention_head_dim=cfg.get("attention_head_dim"),
        mm_rope_freq_dim=cfg.get("mm_rope_freq_dim"),
        mm_rope_theta=cfg.get("mm_rope_theta", 10000.0),
        modality_adaln=cfg.get("modality_adaln", False),
        context_refiner_layers=cfg.get("context_refiner_layers", 0),
        moe_num_experts=cfg.get("moe_num_experts", 0),
        moe_top_k=cfg.get("moe_top_k", 2),
        moe_shared_expert=cfg.get("moe_shared_expert", True),
    ).to(device)


# ---- public builders (take the full project config) ----

def build_text_encoder(config, device):
    return _build_text_encoder(config["text_encoder"], device)


def build_video_vae(config, device):
    return _build_video_vae(config["video_vae"], device)


def build_audio_vae(config, device):
    return _build_audio_vae(config["audio_vae"], device)


def build_dit(config, device, *, video_in_channels, text_in_channels, audio_in_channels=None):
    return _build_dit(
        config["model"], device,
        video_in_channels=video_in_channels,
        text_in_channels=text_in_channels,
        audio_in_channels=audio_in_channels,
    )
