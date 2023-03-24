import torch

from whisper.whisper.audio import N_SAMPLES, N_FRAMES, load_audio
from whisper.whisper.transcribe import log_mel_spectrogram, pad_or_trim


def detect_language(model, audio_path: str):
    audio = load_audio(audio_path)

    mel = log_mel_spectrogram(audio, padding=N_SAMPLES)

    mel_segment = pad_or_trim(mel, N_FRAMES).to(model.device).to(torch.float16)

    _, probs = model.detect_language(mel_segment)
    return max(probs, key=probs.get)
