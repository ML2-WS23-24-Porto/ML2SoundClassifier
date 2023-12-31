import numpy as np
import matplotlib.pyplot as plt
import soundata
import librosa
from librosa import display
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tqdm
import os
import skimage

# for downloading the dataset, put the sound dataset in the same folder as the file after downloading
def download():
    dataset = soundata.initialize('urbansound8k')
    dataset.download()  # download the dataset
    dataset.validate()  # validate that all the expected files are there

def data_preprocess(y,sr,target_sr = 16000): #here we should perform noise reduction , zero padding and resampling,...?
    y_resampled = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    # zero padding, duplicating sound
    if 4 * target_sr//len(y_resampled) >1:
        sig_multiply = y_resampled
        for i in range(4 * target_sr//len(y_resampled)-1): # this copies the short sound, adds noise and appends it to the other sound
            sig_multiply = np.concatenate((sig_multiply,add_noise(y_resampled)), axis=0)
        #print(f"file hase been multiplied by {4 * target_sr//len(y_resampled) }! check {path}")
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

# Here we extract the different spectograms
def get_mfcc(y,dict): # this function gets the mfcc
    # computes the MFCCs
    S = librosa.feature.mfcc(y=y,sr=dict["sr"],n_mfcc=dict["n_mfcc"], hop_length=dict["hop_length"],n_fft=dict["n_fft"],win_length = dict["win_length"])
    return S
def get_chroma_stft(y,dict):
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=dict["sr"], n_chroma=dict["n_chroma"], hop_length=dict["hop_length"],n_fft=dict["n_fft"])
    return chroma_stft
def get_mel_spec(y,dict): # this function gets the mel spectogram
    S = librosa.feature.melspectrogram(y=y, sr=dict["sr"],n_mels=dict["n_mels"], hop_length=dict["hop_length"],n_fft=dict["n_fft"],win_length = dict["win_length"])
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

def scale_minmax(X, min=0.0, max=1.0): # scale
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled

def get_features(S): # to get the features for the data frame
    mean = [np.mean(S.T,axis=0)]
    median = [np.median(S.T,axis=0)]
    max = [np.max(S.T,axis=0)]
    std = [np.std(S.T, axis=0)]
    return mean, max, median, std

def visualize(S,sr,clip_info): # for visualizing the spectograms
    fig, ax = plt.subplots()
    img = librosa.display.specshow(S, x_axis='time', y_axis='mel', sr=sr, fmax=8000, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title=f'Clip {clip_info}')
    plt.show()


def main_loop(metadata,dict, datahome):
    #  dict is a dictionary with all the calc parameters
    print("Processing " +str(len(metadata)) + " files")
    # create dataframes
    df = pd.DataFrame(columns=["slice_file_name","label","labelID","fold", "mean_mfcc", "mean_melspec","max_melspec","max_mfcc","median_melspec","median_mfcc","std_melspec","std_mfcc"])

    for i in tqdm.tqdm(range(len(metadata))):
        filename = str(datahome) + '/audio/fold' + str(metadata["fold"][i]) + '/' + metadata["slice_file_name"][i]
        (sig, rate) = librosa.load(filename, sr=None,res_type="kaiser_fast")
        sig_clean = data_preprocess(y=sig,sr=rate,target_sr=dict["sr"])
        # computes the MFCCs and Melspecs
        mfcc = get_mfcc(sig_clean,dict)
        melspec = get_mel_spec(sig_clean,dict)
        # Turns these to feature vectors
        mean_mfcc, max_mfcc,median_mfcc, std_mfcc = get_features(mfcc)
        mean_melspec, max_melspec, median_melspec, std_melspec = get_features(melspec)
        #save features and metadata in pd Dataframe. Also save images of the Melspec and MFCC in folder for image ML
        df = pd.concat([df,pd.DataFrame({"slice_file_name":metadata["slice_file_name"][i],"label":metadata["class"][i],"labelID":metadata["classID"][i],
                                         "fold":metadata["fold"][i],"mean_mfcc":mean_mfcc,"mean_melspec":mean_melspec,"max_melspec":max_melspec,"max_mfcc":max_mfcc,
                                        "median_melspec":median_melspec,"median_mfcc":median_mfcc, "std_melspec":std_melspec,"std_mfcc":std_mfcc},index=[0])],ignore_index=True)
        save_array(mfcc,output_folder_type="mfcc",fold=metadata["fold"][i],filename=metadata["slice_file_name"][i])
        save_array(melspec,output_folder_type="melspec",fold=metadata["fold"][i],filename=metadata["slice_file_name"][i])
        put_together_save(melspec,mfcc,output_folder_type="both",fold=metadata["fold"][i],filename=metadata["slice_file_name"][i])
        if i%30 == 0: #backup
            df.to_csv("processed_data.csv", index=False)
    df.to_csv("processed_data.csv", index=False)


def process_example(clip_nr,dict,datahome):
    dataset = soundata.initialize(dataset_name='urbansound8k', data_home= datahome)
    ids = dataset.clip_ids  # the list of urbansound8k's clip ids
    clips = dataset.load_clips()
    example_clip = clips[ids[clip_nr]]  # Get clip
    clip_info = example_clip.slice_file_name
    y, rate = example_clip.audio
    librosa.display.waveshow(y=y,sr=rate)
    plt.show()
    print(clip_info)
    sig_clean = data_preprocess(y, rate, target_sr=dict["sr"])
    librosa.display.waveshow(y=sig_clean, sr=dict["sr"])
    plt.show()
    mfcc = get_mfcc(sig_clean, dict)
    melspec = get_mel_spec(sig_clean, dict)
    visualize(mfcc,dict["sr"],clip_info=clip_info)
    visualize(melspec,dict["sr"],clip_info=clip_info)
    img = scale_minmax(mfcc, 0, 255).astype(np.float32)
    img = np.flip(img, axis=0)
    #visualize(chroma_stft,dict["sr"],clip_info=clip_info)


def save_array(array, output_folder_type, fold,filename):
    dir = "sound_datasets/urbansound8k/" + str(output_folder_type) + "/fold" + str(fold)
    # Ensure the array values are in the valid range for an image (0 to 255)
    img = scale_minmax(array, 0, 255).astype(np.uint8)
    img = np.flip(img, axis=0)  # put low frequencies at the bottom in image
    img = 255 - img  # invert. make black==more energy
    # Create the output folder if it doesn't exist
    os.makedirs(dir, exist_ok=True)
    # Replace '.wav' with '.png'
    filename = filename.replace(".wav", ".png")
    # Construct the full path for saving the JPEG file in the 'melspec' folder
    full_path = os.path.join(dir, filename)
    # Save the array as a PNG image
    skimage.io.imsave(full_path, img)

def put_together_save(array1,array2, output_folder_type, fold,filename):
    dir = "sound_datasets/urbansound8k/" + str(output_folder_type) + "/fold" + str(fold)
    # Ensure the array values are in the valid range for an image (0 to 255)
    img1 = scale_minmax(array1, 0, 255).astype(np.uint8)
    img1 = np.flip(img1, axis=0)  # put low frequencies at the bottom in image
    img1 = 255 - img1  # invert. make black==more energy
    img2 = scale_minmax(array2, 0, 255).astype(np.uint8)
    img2 = np.flip(img2, axis=0)  # put low frequencies at the bottom in image
    img2 = 255 - img2  # invert. make black==more energy
    # Create the output folder if it doesn't exist
    img_both = np.vstack((img1, img2))
    os.makedirs(dir, exist_ok=True)
    # Replace '.wav' with '.png'
    filename = filename.replace(".wav", ".png")
    # Construct the full path for saving the JPEG file in the 'melspec' folder
    full_path = os.path.join(dir, filename)
    # Save the array as a PNG image
    skimage.io.imsave(full_path, img_both)


# main loop for data prep:
if __name__== "__main__":
    # for downloading dataset
    #download()
    # define parameters for data extraction
    dict = {}
    # MFCC parameters
    dict["sr"] = 22500
    dict["n_mfcc"] = 36
    dict["hop_length"] = round(dict["sr"] * 0.125)
    dict["win_length"] = round(dict["sr"] * 0.023)
    dict["time_size"] = round(4 * dict["sr"] // dict["hop_length"]) + 1
    # MelSpec parameters
    dict["n_fft"] = 2**13# Window length of fft
    dict["n_mels"] = 40
    dict["fmax"] = round(dict["sr"]/2)

    # File locations
    metadata = pd.read_csv('sound_datasets/urbansound8k/metadata/UrbanSound8K.csv')
    datahome = "sound_datasets/urbansound8k"
    # here you can decide to only process one example or the whole dataset
    process_example(10,dict, datahome)
    #main_loop(metadata,dict,datahome)

