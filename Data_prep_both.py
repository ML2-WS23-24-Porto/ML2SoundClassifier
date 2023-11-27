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



def get_mfcc(y,sr): # this function gets the mfcc
    # MFCC parameters
    n_mfcc = 40
    hop_length = round(target_sr * 0.0125)
    win_length = round(target_sr * 0.023)
    n_fft = 2 ** 14
    mfcc_time_size = 4 * target_sr // hop_length + 1
    # computes the MFCCs
    S = librosa.feature.mfcc(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length,
                                    win_length=win_length, n_mfcc=n_mfcc)
    return S


def get_mel_spec(y,sr): # this function gets the mel spectogram
    n_fft = 2 ** 14
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000,n_fft=n_fft)
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


# main loop for data prep:
if __name__== "__main__":
    # Metadata
    metadata = pd.read_csv('sound_datasets/urbansound8k/metadata/UrbanSound8K.csv')
    metadata.head(10)
    # print(metadata)
    dataset = soundata.initialize(dataset_name='urbansound8k', data_home=r"sound_datasets/urbansound8k")
    ids = dataset.clip_ids  # the list of urbansound8k's clip ids
    clips = dataset.load_clips()  # Load all clips in the dataset
    example_clip = clips[ids[0]]  # Get the first clip
    clip_info = example_clip.slice_file_name
    y, sr = example_clip.audio
    print(clip_info)





    # example processing
    target_sr = 16000
    sig = data_cleaning(y,sr,target_sr=target_sr)
    sig_mfcc = get_mfcc(sig,target_sr)
    sig_melspec = get_mel_spec(sig,target_sr)


    visualize(sig_mfcc,target_sr)
    visualize(sig_melspec,target_sr)




    #for index_num, row in tqdm.tqdm(df.iterrows(), total=len(df)):
    #    file = os.path.join(
    #        os.path.abspath('/kaggle/input/urbansound8k/'), "fold" + str(row["fold"]) + "/",
    #        str(row["slice_file_name"]),
    #    )
    #    label = row["class"]


# after this ml model
