#!pip install librosa
import os
import glob
import random
import numpy as np
import pandas as pd
import scipy.stats


import librosa as ls
import seaborn as sns
import soundfile as sf
import matplotlib.pyplot as plt


from collections import Counter
from utils.augument import consecutive_crop, add_gaussian_noise
from warnings import filterwarnings
filterwarnings('ignore')

def load_audio(file_path):
    y, sr = ls.load(file_path, sr=None)
    return y, sr

def add_shared_colorbar(fig, img, label=None):
    cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # right side
    cbar = fig.colorbar(img, cax=cax)

    if label:
        cbar.set_label(label)

    return cbar

def sample_one_file_per_genre(DATA_PATH, genres, exts=('.wav', '.mp3', '.au')):
    samples = {}

    for genre in genres:
        genre_dir = os.path.join(DATA_PATH, genre)
        files = [f for f in os.listdir(genre_dir) if f.endswith(exts)]

        if files:
            samples[genre] = os.path.join(genre_dir, random.choice(files))
        else:
            samples[genre] = None

    return samples

def plot_waveform_grid(samples, n_rows=2, n_cols=5):
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 6))
    axes = axes.flatten()

    for ax, (genre, file_path) in zip(axes, samples.items()):
        if file_path is None:
            ax.set_title(f"{genre}\n(No file)")
            ax.axis("off")
            continue

        y, sr = load_audio(file_path)
        ls.display.waveshow(y, sr=sr, ax=ax)
        ax.set_title(genre)
        ax.set_ylabel("Amp")

    fig.suptitle("Waveform Comparison Across Genres", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.show()

def plot_spectrogram_grid(samples, n_rows=2, n_cols=5):
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 6))
    axes = axes.flatten()

    for ax, (genre, file_path) in zip(axes, samples.items()):
        if file_path is None:
            ax.axis("off")
            continue

        y, sr = load_audio(file_path)
        D = ls.stft(y)
        S_db = ls.amplitude_to_db(np.abs(D), ref=np.max)

        img = ls.display.specshow(
            S_db, sr=sr, x_axis='time', y_axis='log', ax=ax, cmap='magma'
        )
        ax.set_title(genre)

    fig.suptitle("Log-Frequency Spectrogram Comparison", fontsize=16)

    add_shared_colorbar(fig, img, label="dB")

    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.show()

def plot_mel_grid(samples, n_rows=2, n_cols=5):
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 6))
    axes = axes.flatten()

    for ax, (genre, file_path) in zip(axes, samples.items()):
        if file_path is None:
            ax.axis("off")
            continue

        y, sr = load_audio(file_path)
        M = ls.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        M_db = ls.power_to_db(M, ref=np.max)

        img = ls.display.specshow(
            M_db, sr=sr, x_axis='time', y_axis='mel', ax=ax, cmap='coolwarm'
        )
        ax.set_title(genre)

    fig.suptitle("Mel-Spectrogram Comparison (DL-friendly)", fontsize=16, fontweight="bold")
    add_shared_colorbar(fig, img, label="dB")

    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.show()

def plot_chroma_grid(samples, n_rows=2, n_cols=5):
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 6))
    axes = axes.flatten()

    for ax, (genre, file_path) in zip(axes, samples.items()):
        if file_path is None:
            ax.axis("off")
            continue

        y, sr = load_audio(file_path)
        C = ls.feature.chroma_cqt(y=y, sr=sr)

        img = ls.display.specshow(
            C, x_axis='time', y_axis='chroma', ax=ax, cmap='viridis'
        )
        ax.set_title(genre)

    fig.suptitle("Chromagram (Harmony / Pitch Class)", fontsize=16, fontweight="bold")
    add_shared_colorbar(fig, img, label="Energy")
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.show()


def visualize_genre_dashboard(file_path, genre_name):
    """
    Plot 4 features of 1 random sample for each class
    """
    # 1. Load Audio
    y, sr = ls.load(file_path, sr=None)

    # 2. Tính toán các Features
    # -- STFT Spectrogram (Log scale)
    D = ls.stft(y)
    S_db = ls.amplitude_to_db(np.abs(D), ref=np.max)

    # -- Mel Spectrogram (Mô phỏng tai người)
    M = ls.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    M_db = ls.power_to_db(M, ref=np.max)

    # Dùng CQT (Constant-Q transform) để hiển thị nốt nhạc đẹp và rõ hơn STFT thường
    C = ls.feature.chroma_cqt(y=y, sr=sr)

    # 3. Setup khung hình (Dashboard)
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
    fig.suptitle(f'Genre: {genre_name.upper()} | File: {os.path.basename(file_path)}', fontsize=16, fontweight='bold')

    # --- Plot 1: Waveform (Miền thời gian) ---
    ls.display.waveshow(y, sr=sr, ax=axes[0, 0], color='blue', alpha=0.6)
    axes[0, 0].set_title('1. Waveform (Time Domain)')
    axes[0, 0].set_ylabel('Amplitude')

    # --- Plot 2: Log-Frequency Spectrogram ---
    img2 = ls.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log', ax=axes[0, 1], cmap='magma')
    axes[0, 1].set_title('2. Spectrogram (Log Frequency)')
    fig.colorbar(img2, ax=axes[0, 1], format='%+2.0f dB')

    # --- Plot 3: Mel-Spectrogram (Tai người / Deep Learning Input) ---
    img3 = ls.display.specshow(M_db, sr=sr, x_axis='time', y_axis='mel', ax=axes[1, 0], cmap='coolwarm')
    axes[1, 0].set_title('3. Mel-Spectrogram (Mel Frequency)')
    fig.colorbar(img3, ax=axes[1, 0], format='%+2.0f dB')

    # --- Plot 4: Chromagram (Harmonic / Pitch Class) ---
    img4 = ls.display.specshow(C, sr=sr, x_axis='time', y_axis='chroma', ax=axes[1, 1], cmap='viridis')
    axes[1, 1].set_title('4. Chromagram (Pitch/Harmony)')
    fig.colorbar(img4, ax=axes[1, 1])

    plt.tight_layout()
    plt.show()


def processing_image(data_path):
    lst_type = ['waveform', 'spectrogram', 'mel_spectrogram', 'chromagram']
    base_out = 'Data/images_data'
    os.makedirs(base_out, exist_ok=True)

    for typ in lst_type:
        for genres in os.listdir(data_path):
            out_dir = f"{base_out}/{typ}/{genres}"
            os.makedirs(out_dir, exist_ok=True)

            for i, file in enumerate(glob.glob(os.path.join(data_path, genres, '*.wav'))):
                y, sr = ls.load(file, mono=True, sr=22050)

                plt.figure(figsize=(10,6))

                if typ == 'waveform':
                    ls.display.waveshow(y, sr=sr)

                elif typ == 'spectrogram':
                    S = np.abs(ls.stft(y, n_fft=1024, hop_length=512))
                    S_db = ls.amplitude_to_db(S, ref=np.max)
                    ls.display.specshow(S_db, sr=sr)

                elif typ == 'mel_spectrogram':
                    M = ls.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=512)
                    M_db = ls.power_to_db(M, ref=np.max)
                    ls.display.specshow(M_db, sr=sr)

                elif typ == 'chromagram':
                    C = ls.feature.chroma_stft(y=y, sr=sr, n_fft=1024, hop_length=512)
                    ls.display.specshow(C, sr=sr)

                plt.axis('off')
                plt.savefig(f"{out_dir}/img_{i}.png", bbox_inches='tight', pad_inches=0)
                plt.clf()
                plt.close('all')



def visualize_consecutive_crops(
    audio_path,
    duration=3.0,
    n_rows=5,
    n_cols=2
):
    y, sr = ls.load(audio_path, sr=None, mono=True)

    crop_len = int(sr * duration)
    max_segments = (n_rows * n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 10))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        start = i * crop_len
        end = start + crop_len

        if end > len(y):
            ax.axis("off")
            continue

        y_crop = y[start:end]
        t = np.linspace(0, duration, len(y_crop))

        ax.plot(t, y_crop, linewidth=0.8)
        ax.set_title(f"Segment {i+1} ({i*duration:.1f}s–{(i+1)*duration:.1f}s)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def visualize_noise_effect_consecutive(
    audio_path,
    duration=3.0,
    noise_level="small",
    segment_idx=0
):
    y, sr = ls.load(audio_path, sr=None, mono=True)

    crop_len = int(sr * duration)
    start = segment_idx * crop_len
    end = start + crop_len

    if end > len(y):
        raise ValueError("Segment index out of range")

    y_crop = y[start:end]
    y_noisy = add_gaussian_noise(y_crop, noise_level)

    # Mel spectrograms
    mel_clean = ls.feature.melspectrogram(y=y_crop, sr=sr)
    mel_noisy = ls.feature.melspectrogram(y=y_noisy, sr=sr)

    mel_clean_db = ls.power_to_db(mel_clean, ref=np.max)
    mel_noisy_db = ls.power_to_db(mel_noisy, ref=np.max)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # Waveforms
    axes[0, 0].plot(y_crop, linewidth=0.8)
    axes[0, 0].set_title("Waveform (Clean)")

    axes[0, 1].plot(y_noisy, linewidth=0.8)
    axes[0, 1].set_title("Waveform (With Gaussian Noise)")

    # Mel spectrograms (NO colorbar để khỏi đè)
    ls.display.specshow(
        mel_clean_db,
        sr=sr,
        x_axis="time",
        y_axis="mel",
        ax=axes[1, 0]
    )
    axes[1, 0].set_title("Mel Spectrogram (Clean)")

    ls.display.specshow(
        mel_noisy_db,
        sr=sr,
        x_axis="time",
        y_axis="mel",
        ax=axes[1, 1]
    )
    axes[1, 1].set_title("Mel Spectrogram (With Gaussian Noise)")

    plt.tight_layout()
    plt.show()


def extract_features(y, sr):
    duration = len(y) / sr

    # =========================
    # 1. TIME-DOMAIN
    # =========================
    zcr = ls.feature.zero_crossing_rate(y)[0]
    rms = ls.feature.rms(y=y)[0]

    y_harm, y_perc = ls.effects.hpss(y)


    # Silence ratio
    intervals = ls.effects.split(y, top_db=40)
    if len(intervals) > 0:
        active = np.sum(intervals[:, 1] - intervals[:, 0]) / sr
        silence_ratio = 1 - active / duration
    else:
        silence_ratio = 1.0

    try:
        onset_env = ls.onset.onset_strength(y=y, sr=sr)
        tempo = ls.feature.tempo(onset_envelope=onset_env, sr=sr)[0]
    except:
        tempo = 0

    # =========================
    # 2. SPECTRAL FEATURES
    # =========================
    S = np.abs(ls.stft(y))

    centroid = ls.feature.spectral_centroid(S=S, sr=sr)[0]
    bandwidth = ls.feature.spectral_bandwidth(S=S, sr=sr)[0]
    rolloff = ls.feature.spectral_rolloff(S=S, sr=sr)[0]
    contrast = ls.feature.spectral_contrast(S=S, sr=sr)

    # --- MỚI: Spectral Flatness ---
    flatness = ls.feature.spectral_flatness(y=y)[0]

    mfcc = ls.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_delta = ls.feature.delta(mfcc)
    chroma = ls.feature.chroma_stft(S=S, sr=sr)

    # =========================
    # 3. AGGREGATION (Mean & Std)
    # =========================
    feats = {
        "duration": duration,
        "silence_ratio": silence_ratio,
        "tempo": tempo,

        "zcr_mean": np.mean(zcr), "zcr_std": np.std(zcr),
        "rms_mean": np.mean(rms), "rms_std": np.std(rms),

        # New features aggregation
        "harm_mean": np.mean(y_harm), "harm_var": np.var(y_harm),
        "perc_mean": np.mean(y_perc), "perc_var": np.var(y_perc),
        "flatness_mean": np.mean(flatness), "flatness_var": np.var(flatness),

        "centroid_mean": np.mean(centroid), "centroid_std": np.std(centroid),
        "bandwidth_mean": np.mean(bandwidth), "bandwidth_std": np.std(bandwidth),
        "rolloff_mean": np.mean(rolloff), "rolloff_std": np.std(rolloff)
    }

    # MFCC (20) & Delta (20)
    for i in range(20):
        feats[f"mfcc{i+1}_mean"] = np.mean(mfcc[i])
        feats[f"mfcc{i+1}_std"]  = np.std(mfcc[i])
        feats[f"mfcc_delta{i+1}_mean"] = np.mean(mfcc_delta[i])
        feats[f"mfcc_delta{i+1}_std"]  = np.std(mfcc_delta[i])

    # Chroma (12)
    for i in range(12):
        feats[f"chroma{i+1}_mean"] = np.mean(chroma[i])
        feats[f"chroma{i+1}_std"]  = np.std(chroma[i])

    # Contrast (7 bands)
    for i in range(contrast.shape[0]):
        feats[f"contrast{i+1}_mean"] = np.mean(contrast[i])
        feats[f"contrast{i+1}_std"]  = np.std(contrast[i])

    return feats