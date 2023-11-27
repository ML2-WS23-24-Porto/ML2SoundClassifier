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

# for downloading the dataset, put the sound dataset in the same folder as the file after downloading
#dataset = soundata.initialize('urbansound8k')
#dataset.download()  # download the dataset
#dataset.validate()  # validate that all the expected files are there

#example_clip = dataset.choice_clip()  # choose a random example clip
#print(example_clip)



def data_cleaning(y,sr,target_sr = 16000): #here we should perform noise reduction , zero padding and resampling,...?
    y_resampled = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    # zero padding
    if len(y_resampled) < 4 * target_sr:
        sig = np.concatenate((y_resampled, np.zeros(4 * target_sr - len(y_resampled))), axis=0)
    else:
        sig = y_resampled
    return sig



def get_mfcc(y,sr,n_mfcc,hop_length,win_length,n_fft=2**14): # this function gets the mfcc
    # computes the MFCCs
    S = librosa.feature.mfcc(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length,
                                    win_length=win_length, n_mfcc=n_mfcc)
    return S


def get_mel_spec(y,sr,n_mels,n_fft=2**14,fmax = 8000): # this function gets the mel spectogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=fmax,n_fft=n_fft)
    S_dB = librosa.power_to_db(S, ref=np.max)
    return S_dB

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


def visualize(S,sr):
    fig, ax = plt.subplots()
    img = librosa.display.specshow(S, x_axis='time', y_axis='mel', sr=sr, fmax=8000, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title=f'Mel-frequency spectrogram of clip {clip_info} with samplerate {sr} Hz')
    plt.show()


def main(files,sr):
    # MFCC parameters
    n_mfcc = 40
    hop_length = round(sr * 0.0125)
    win_length = round(sr * 0.023)
    n_fft = 2**14
    mfcc_time_size = 4 * sr // hop_length + 1
    # MelSpec parameters
    n_fft = 2 ** 14
    n_mels = 128
    fmax = 8000
    # create dataframes
    # read all wav file without resampling
    dataset = np.zeros(shape=[len(files), 4 * sr])
    dataset_mfcc = np.zeros(shape=[len(files), n_mfcc, mfcc_time_size])
    dataset_melspec = np.zeros(shape=[len(files), n_mfcc, mfcc_time_size])
    # example processing
    i = 0
    for f in files:
        (sig, rate) = librosa.load(f, sr=None)
        sig_clean = data_cleaning(y=sig,sr=rate,target_sr=sr)
        dataset[i] = sig_clean
        # computes the MFCCs

        dataset_mfcc[i] = get_mfcc(sig_clean,sr,n_mfcc=n_mfcc,hop_length=hop_length,win_length=win_length,n_fft=n_fft)
        dataset_melspec[i] = get_mel_spec(sig_clean,sr,n_mels=n_mels,n_fft=n_fft,fmax=fmax)
        i += 1




def process_example(clip_nr,target_sr=16000):
    # MFCC parameters
    n_mfcc = 40
    hop_length = round(sr * 0.0125)
    win_length = round(sr * 0.023)
    n_fft = 2 ** 14
    mfcc_time_size = 4 * sr // hop_length + 1
    # MelSpec parameters
    n_fft = 2 ** 14
    n_mels = 128
    fmax = 8000
    example_clip = clips[ids[clip_nr]]  # Get clip
    clip_info = example_clip.slice_file_name
    y, sr = example_clip.audio
    print(clip_info)
    sig = data_cleaning(y, sr, target_sr=target_sr)
    sig_mfcc = get_mfcc(sig_clean,sr,n_mfcc=n_mfcc,hop_length=hop_length,win_length=win_length,n_fft=n_fft)
    sig_melspec = get_mel_spec(sig_clean,sr,n_mels=n_mels,n_fft=n_fft,fmax=fmax)
    visualize(sig_mfcc,target_sr)
    visualize(sig_melspec,target_sr)





# main loop for data prep:
if __name__== "__main__":
    # Metadata
    metadata = pd.read_csv('sound_datasets/urbansound8k/metadata/UrbanSound8K.csv')
    metadata.head(10)
    # print(metadata)
    dataset = soundata.initialize(dataset_name='urbansound8k', data_home=r"sound_datasets/urbansound8k")
    ids = dataset.clip_ids  # the list of urbansound8k's clip ids
    clips = dataset.load_clips()  # Load all clips in the dataset
    #example_clip = clips[ids[0]]  # Get the first clip
    #clip_info = example_clip.slice_file_name
    #y, sr = example_clip.audio







    #for index_num, row in tqdm.tqdm(df.iterrows(), total=len(df)):
    #    file = os.path.join(
    #        os.path.abspath('/kaggle/input/urbansound8k/'), "fold" + str(row["fold"]) + "/",
    #        str(row["slice_file_name"]),
    #    )
    #    label = row["class"]


# after this ml model
