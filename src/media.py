"""Media output helpers: gif / frame strip / wav / mp4 (H.264 + AAC mux)."""

import numpy as np
from PIL import Image


def to_uint8_frames(videos):
    """VAE decode output (B, 3, T, H, W) in [-1, 1] -> uint8 (B, T, H, W, 3)."""
    v = ((videos.float() + 1) / 2).clamp(0, 1)
    return (v.permute(0, 2, 3, 4, 1).cpu().numpy() * 255).astype(np.uint8)


def save_gif(frames, path, fps=8):
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=int(1000 / fps), loop=0)


def save_strip(frames, path, every=2):
    picked = frames[::every]
    strip = np.concatenate(list(picked), axis=1)
    Image.fromarray(strip).save(path)


def save_wav(waveform, path, sample_rate=44100):
    """waveform: float (2, N) in [-1, 1] -> 16-bit stereo PCM."""
    import wave as wave_module

    pcm = (waveform.T * 32767).astype(np.int16)  # (N, 2) interleaved
    with wave_module.open(str(path), "wb") as f:
        f.setnchannels(pcm.shape[1])
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(pcm.tobytes())


def save_mp4(frames, path, wav_path=None, fps=8):
    """Encode frames (T, H, W, 3) uint8 + optional wav into H.264/AAC mp4,
    using the ffmpeg binary bundled with imageio-ffmpeg."""
    import subprocess

    from imageio_ffmpeg import get_ffmpeg_exe

    t, h, w, _ = frames.shape
    cmd = [
        get_ffmpeg_exe(), "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
    ]
    if wav_path is not None:
        # Audio VAEs can decode a few milliseconds shorter than the video grid.
        # Padding before `-shortest` makes the video stream determine the output
        # duration instead of silently dropping the final video frames.
        cmd += ["-i", str(wav_path), "-af", "apad", "-c:a", "aac", "-shortest"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(path)]
    proc = subprocess.run(cmd, input=np.ascontiguousarray(frames).tobytes(), capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.decode()[-500:]}")
