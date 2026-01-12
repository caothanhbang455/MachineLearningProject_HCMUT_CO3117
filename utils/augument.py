import numpy as np
import librosa as ls
import random


def consecutive_crop(y, sr, duration=3.0, segment_idx=0):
    """
    Extract a consecutive fixed-length segment from audio.

    segment_idx: index of the segment (0, 1, 2, ...)
    """
    crop_len = int(sr * duration)
    start = segment_idx * crop_len
    end = start + crop_len

    if end > len(y):
        return None

    return y[start:end]


def add_gaussian_noise(y, level="small"):
    """
    Add low-amplitude Gaussian noise to waveform.
    """
    if level == "small":
        scale = np.random.uniform(0.001, 0.003)
    elif level == "medium":
        scale = np.random.uniform(0.003, 0.008)
    else:
        raise ValueError("level must be 'small' or 'medium'")

    noise = np.random.randn(len(y))
    return y + scale * noise



def augment_audio_consecutive(
    path,
    sr=None,
    crop_duration=3.0,
    segment_idx=0,
    noise_prob=0.5,
    noise_level="small"
):
    """
    Augment audio using consecutive segmentation.

    Pipeline:
        load - consecutive crop - optional noise
    """
    y, sr = ls.load(path, sr=sr, mono=True)

    y_crop = consecutive_crop(y, sr, crop_duration, segment_idx)
    if y_crop is None:
        return None, sr

    if random.random() < noise_prob:
        y_crop = add_gaussian_noise(y_crop, noise_level)

    return y_crop, sr
