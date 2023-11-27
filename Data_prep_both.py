import numpy as np
import matplotlib.pyplot as plt
import soundata
import librosa
from librosa import display
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import tqdm
import os

# for downloading the dataset
#dataset = soundata.initialize('urbansound8k')
#dataset.download()  # download the dataset
#dataset.validate()  # validate that all the expected files are there

#example_clip = dataset.choice_clip()  # choose a random example clip
#print(example_clip)

dataset = soundata.initialize(dataset_name='urbansound8k',data_home="/home/elias88348/sound_datasets/urbansound8k")
ids = dataset.clip_ids  # the list of urbansound8k's clip ids
clips = dataset.load_clips()  # Load all clips in the dataset
example_clip = clips[ids[0]]  # Get the first clip
clip_info = example_clip.slice_file_name
y, sr = example_clip.audio
print(clip_info)

metadata = pd.read_csv('/home/elias88348/sound_datasets/urbansound8k/metadata/UrbanSound8K.csv')
metadata.head(10)
#print(metadata)

def data_cleaning(): #here we should perform noise reduction , zero padding and resampling,...?
    target_sr = 16000
    y_downsampled = librosa.resample(y, sr, target_sr)
    return y_downsampled


def get_mfcc(): # this function gets the mfcc
    (sig, rate) = librosa.load(f, sr=None)
    sig_res = librosa.resample(sig, orig_sr=rate, target_sr=target_sr)
    # zero padding
    if len(sig_res) < 4 * target_sr:
        sig_res_pad = np.concatenate((sig_res, np.zeros(4 * target_sr - len(sig_res))), axis=0)
    else:
        sig_res_pad = sig_res
    dataset[i] = sig_res_pad
    # computes the MFCCs
    sig_mfcc = librosa.feature.mfcc(y=sig_res_pad, sr=target_sr, n_fft=n_fft, hop_length=hop_length,
                                    win_length=win_length, n_mfcc=n_mfcc)
    return

def get_mel_spec(): # this function gets the mel spectogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    fig, ax = plt.subplots()
    S_dB = librosa.power_to_db(S, ref=np.max)
    df = pd.DataFrame(S_dB)
    return

def normalize_spectogram(clip): # bring spectograms in the form we want, suitable for ml models
    # Create a MinMaxScaler
    scaler = MinMaxScaler()
    df = pd.DataFrame(clip)
    # Apply Min-Max Scaling to the DataFrame
    df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    print("Original DataFrame:")
    print(df)
    print("\nNormalized DataFrame:")
    print(df_normalized)

# after this ml model


