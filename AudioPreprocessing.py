import tensorflow as tf
import numpy as np
import pandas as pd
import gc
import os
import copy
from joblib import load, dump
import librosa.display

from tqdm import tqdm
from scipy import signal as sci_signal
import matplotlib.pyplot as plt

import cv2


def spects_from_audios(auds, img_shape = 256):
    '''
        list of 5 scond audio files
    '''
    Spectrograms = []

    for row in (auds):

        mean_signal = np.nanmean(row)
        audio_data = np.nan_to_num(row, nan=mean_signal) if np.isnan(
            row).mean() < 1 else np.zeros_like(row)

        frequencies, times, spec_data = sci_signal.spectrogram(
            audio_data,
            fs=32000,
            nfft=1095,
            nperseg=412,
            noverlap=100,
            window='hann'
        )

        valid_freq = (frequencies >= 100) & (frequencies <= 15000)
        spec_data = spec_data[valid_freq, :]

        spec_data = np.log10(spec_data + 1e-20)

        spec_data = spec_data - spec_data.min()
        spec_data = spec_data / spec_data.max()

        spec_data = cv2.resize(
            spec_data, (img_shape, img_shape), interpolation=cv2.INTER_AREA)
        Spectrograms.append(spec_data)

    Spectrograms = np.asarray(Spectrograms)
    return Spectrograms


def split_audio_file(path):
    sr = 32000
    x = []

    audio_file = path
    y, _ = librosa.load(audio_file, sr=sr)
    total_duration = len(y) / sr

    while total_duration < 5:
        y = np.concatenate([y, y])
        total_duration = len(y) / sr

    start_idx = int((total_duration / 2 - 2.5) * sr)
    middle_5_seconds = y[start_idx:start_idx + 5 * sr]
    early_5_sec = y[0:5*sr]

    x.append(middle_5_seconds)
    x.append(early_5_sec)
    return x
