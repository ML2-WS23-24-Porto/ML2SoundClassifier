import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import soundata
import librosa
from librosa import display
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import tqdm
import os
import imageio
import torch
import scipy

# for downloading the dataset, put the sound dataset in the same folder as the file after downloading
def download():
    dataset = soundata.initialize('urbansound8k')
    dataset.download()  # download the dataset
    dataset.validate()  # validate that all the expected files are there

def data_preprocess(y,sr,target_sr = 16000,path = None): #here we should perform noise reduction , zero padding and resampling,...?
    y_resampled = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    # zero padding, duplicating sound
    if 4 * target_sr//len(y_resampled) >1:
        sig_multiply = y_resampled
        for i in range(4 * target_sr//len(y_resampled)-1): # this copies the short sound, adds noise and appends it to the other sound
            sig_multiply = np.concatenate((sig_multiply,add_noise(y_resampled)), axis=0)
        print(f"file hase been multiplied by {4 * target_sr//len(y_resampled) }! check {path}")
        sig = np.concatenate((sig_multiply, np.zeros(4 * target_sr - len(sig_multiply))), axis=0)
    elif len(y_resampled) < 4 * target_sr:
        sig = np.concatenate((y_resampled, np.zeros(4 * target_sr - len(y_resampled))), axis=0)
    else:
        sig = y_resampled
    return sig

def add_noise(sound_clip):
    noise = np.ones(len(sound_clip))
    noise_amp = np.random.uniform(0.005, 0.008,size=len(sound_clip))
    noisy_sound_clip = sound_clip + (noise_amp * noise)
    return noisy_sound_clip

def get_mfcc(y,sr,n_mfcc,hop_length,win_length,n_fft=2**14): # this function gets the mfcc
    # computes the MFCCs
    S = librosa.feature.mfcc(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length,
                                    win_length=win_length, n_mfcc=n_mfcc)
    return S

def get_mel_spec(y,sr,n_mels,hop_length,n_fft=2**14,fmax = 8000): # this function gets the mel spectogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels,hop_length=hop_length, fmax=fmax,n_fft=n_fft)
    S_dB = librosa.power_to_db(S, ref=np.max)
    return S_dB

def get_chroma(y,sr,n_chroma):
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=n_chroma).T, axis=0)
    return chroma
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

def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled


def visualize(S,sr,clip_info):
    fig, ax = plt.subplots()
    img = librosa.display.specshow(S, x_axis='time', y_axis='mel', sr=sr, fmax=8000, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title=f'Mel-frequency spectrogram of clip {clip_info} with samplerate {sr} Hz')
    plt.show()


def main_loop(metadata,dict):
    #  dict is a dictionary with all the calc parameters
    # create dataframes
    # read all wav file without resampling
    dataset = np.zeros(shape=[len(metadata), 4 * dict["sr"]])
    dataset_mfcc = np.zeros(shape=[len(metadata), dict["n_mfcc"], dict["time_size"]])
    dataset_melspec = np.zeros(shape=[len(metadata), dict["n_mels"], dict["time_size"]])
    # example processing
    i=0
    print(len(metadata))

    df = pd.DataFrame(columns=["slice_file_name","label","labelID","fold", "f_mfcc", "f_melspec"])

    for i in range(len(metadata)):
        filename = 'sound_datasets/urbansound8k/audio/fold' + str(metadata["fold"][i]) + '/' + metadata["slice_file_name"][i]
        (sig, rate) = librosa.load(filename, sr=None,res_type="kaiser_fast")
        sig_clean = data_preprocess(y=sig,sr=rate,target_sr=dict["sr"],path=filename)
        dataset[i] = sig_clean
        # computes the MFCCs
        dataset_mfcc[i] = get_mfcc(sig_clean,dict["sr"],n_mfcc=dict["n_mfcc"],hop_length=dict["hop_length"],win_length=dict["win_length"],n_fft=dict["n_fft"])
        dataset_melspec[i] = get_mel_spec(sig_clean,dict["sr"],n_mels=dict["n_mels"],hop_length=dict["hop_length"],n_fft=dict["n_fft"],fmax=dict["fmax"])
        feature_mfcc = [np.mean(dataset_mfcc[i].T,axis=0)]
        feature_melspec = [np.mean(dataset_melspec[i].T,axis=0)]
        df = pd.concat([df,pd.DataFrame({"slice_file_name":metadata["slice_file_name"][i],"label":metadata["class"][i],"labelID":metadata["classID"][i],
                                         "fold":metadata["fold"][i],"f_mfcc":feature_mfcc,"f_melspec":feature_melspec},index=[0])],ignore_index=True)
        save_array_as_jpeg(dataset_mfcc[i],output_folder_type="mfcc",fold=metadata["fold"][i],filename=metadata["slice_file_name"][i])
        save_array_as_jpeg(dataset_melspec[i],output_folder_type="melspec",fold=metadata["fold"][i],filename=metadata["slice_file_name"][i])
        print(f"prepared file{i} from {len(metadata)}!")
        i += 1
        if i%30 == 0: #backup
            df.to_csv("processed_data.csv", index=False)




def process_example(clip_nr,sr=16000):
    dataset = soundata.initialize(dataset_name='urbansound8k', data_home="sound_datasets/urbansound8k")
    ids = dataset.clip_ids  # the list of urbansound8k's clip ids
    clips = dataset.load_clips()
    # MFCC parameters
    n_mfcc = 40
    hop_length = round(sr * 0.0125)
    win_length = round(sr * 0.023)
    mfcc_time_size = 4 * sr // hop_length + 1
    # MelSpec parameters
    n_fft = 2 ** 14
    n_mels = 128
    fmax = 8000
    example_clip = clips[ids[clip_nr]]  # Get clip
    clip_info = example_clip.slice_file_name
    y, rate = example_clip.audio
    librosa.display.waveshow(y=y,sr=rate)
    plt.show()
    print(clip_info)
    sig_clean = data_preprocess(y, rate, target_sr=sr)
    librosa.display.waveshow(y=sig_clean, sr=sr)
    plt.show()
    sig_mfcc = get_mfcc(sig_clean,sr,n_mfcc=n_mfcc,hop_length=hop_length,win_length=win_length,n_fft=n_fft)
    sig_melspec = get_mel_spec(sig_clean,sr,n_mels=n_mels,hop_length=hop_length,n_fft=n_fft,fmax=fmax)
    visualize(sig_mfcc,sr,clip_info=clip_info)
    visualize(sig_melspec,sr,clip_info=clip_info)


def save_array_as_jpeg(array, output_folder_type, fold,filename):
    dir = "sound_datasets/urbansound8k/" + str(output_folder_type) + "/fold" + str(fold)
    # Ensure the array values are in the valid range for an image (0 to 255)
    img = scale_minmax(array, 0, 255).astype(np.uint8)
    img = np.flip(img, axis=0)  # put low frequencies at the bottom in image
    img = 255 - img  # invert. make black==more energy
    # Create the output folder if it doesn't exist
    os.makedirs(dir, exist_ok=True)
    # Replace '.wav' with '.jpeg'
    filename = filename.replace(".wav", ".jpeg")
    # Construct the full path for saving the JPEG file in the 'melspec' folder
    full_path = os.path.join(dir, filename)
    # Save the array as a JPEG image
    matplotlib.image.imsave(full_path, img)



# main loop for data prep:
if __name__== "__main__":
    dict = {}
    # MFCC parameters
    dict["sr"]=16000
    dict["n_mfcc"] = 40
    dict["hop_length"] = round(dict["sr"] * 0.025)
    dict["win_length"] = round(dict["sr"] * 0.023)
    dict["time_size"] = 4 * dict["sr"] // dict["hop_length"] + 1
    # MelSpec parameters
    dict["n_fft"] = 2 ** 14
    dict["n_mels"] = 128
    dict["fmax"] = 8000

    # Metadata
    metadata = pd.read_csv('sound_datasets/urbansound8k/metadata/UrbanSound8K.csv')
    metadata.head(10)
    print(metadata)
    #process_example(20,sr)
    main_loop(metadata,dict)
