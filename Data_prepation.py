import numpy as np
import matplotlib.pyplot as plt
import soundata
import librosa
from librosa import display
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# audio: The clip's audio
  #           * np.ndarray - audio signal
  #           * float - sample rate,
  # class_id: The clip's class id.
  #           * int - integer representation of the class label (0-9). See Dataset Info in the documentation for mapping,
  # class_label: The clip's class label.
  #           * str - string class name: air_conditioner, car_horn, children_playing, dog_bark, drilling, engine_idling, gun_shot, jackhammer, siren, street_music,
  # fold: The clip's fold.
  #           * int - fold number (1-10) to which this clip is allocated. Use these folds for cross validation,
  # freesound_end_time: The clip's end time in Freesound.
  #           * float - end time in seconds of the clip in the original freesound recording,
  # freesound_id: The clip's Freesound ID.
  #           * str - ID of the freesound.org recording from which this clip was taken,
  # freesound_start_time: The clip's start time in Freesound.
  #           * float - start time in seconds of the clip in the original freesound recording,
  # salience: The clip's salience.
  #           * int - annotator estimate of class sailence in the clip: 1 = foreground, 2 = background,
  # slice_file_name: The clip's slice filename.
  #           * str - The name of the audio file. The name takes the following format: [fsID]-[classID]-[occurrenceID]-[sliceID].wav,
  # tags: The clip's tags.
  #           * annotations.Tags - tag (label) of the clip + confidence. In UrbanSound8K every clip has one tag,
  #


dataset = soundata.initialize(dataset_name='urbansound8k',data_home=r"C:\Users\Diederik\OneDrive\Bureaublad\studie tn\Minor vakken Porto\Machine Learning\UrbanSound8K")
ids = dataset.clip_ids  # the list of urbansound8k's clip ids
clips = dataset.load_clips()  # Load all clips in the dataset
example_clip = clips[ids[0]]  # Get the first clip
clip_info = example_clip.slice_file_name
y, sr = example_clip.audio
print(clip_info)
#print(dataset.choice_clip())

#downsampled version
target_sr = 16000
y_downsampled = librosa.resample(y, sr, target_sr)


#ideas for data formatting: mel spectograms and WaveNet and Similar Architectures:

#mel spectogram:
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,fmax=8000)
fig, ax = plt.subplots()
S_dB = librosa.power_to_db(S, ref=np.max)
df = pd.DataFrame(S_dB)
print(df)
img = librosa.display.specshow(S_dB, x_axis='time',y_axis='mel', sr=sr,fmax=8000, ax=ax)
fig.colorbar(img, ax=ax, format='%+2.0f dB')
ax.set(title=f'Mel-frequency spectrogram of clip {clip_info} with samplerate {sr} Hz')
plt.show()

#mel spectogram downasampled vesion
S_downsampled = librosa.feature.melspectrogram(y=y_downsampled, sr=target_sr, n_mels=128,fmax=8000)
fig, ax = plt.subplots()
S_dB = librosa.power_to_db(S_downsampled, ref=np.max)
df = pd.DataFrame(S_dB)
print(df)
img = librosa.display.specshow(S_dB, x_axis='time',y_axis='mel', sr=sr,fmax=8000, ax=ax)
fig.colorbar(img, ax=ax, format='%+2.0f dB')
ax.set(title=f'Mel-frequency spectrogram of clip {clip_info} with samplerate {target_sr} Hz')
plt.show()


def normalize_spectogram(clip):
    # Create a MinMaxScaler
    scaler = MinMaxScaler()
    df = pd.DataFrame(clip)
    # Apply Min-Max Scaling to the DataFrame
    df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    print("Original DataFrame:")
    print(df)
    print("\nNormalized DataFrame:")
    print(df_normalized)

clip_norm = normalize_spectogram(S_dB)




