import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import os
import IPython.display as ipd
import tensorflow
import pickle

print("This is the starting cwd:")
print(os.getcwd())

#collecting the dataset
dataset_path = "Data/genres_original"
metadata = pd.read_csv("Data/features_30_sec.csv")

def features_extractor(file):
    audio, sample_rate = librosa.load(file, res_type='kaiser_fast')
    # extract mfccs
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled = np.mean(mfccs,axis=1)
    # extract chroma_stfts
    # chroma_stft = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
    # chroma_stft_scaled = np.mean(chroma_stft,axis=1)
    # extract spectral_centroid
    # spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
    # spectral_centroid_scaled = np.mean(spectral_centroid,axis=1)
    # extract spectral_bandwidth
    # spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate)
    # spectral_bandwidth_scaled = np.mean(spectral_bandwidth,axis=1)
    # extract zero_crossing_rate
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=audio)
    zero_crossing_rate_scaled = np.mean(zero_crossing_rate,axis=1)
    # extract rms
    rms = librosa.feature.rms(y=audio)
    rms_scaled = np.mean(rms,axis=1)
    # extract tempo
    onset_env = librosa.onset.onset_strength(y=audio, sr=sample_rate)
    tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sample_rate)
    
    # combine features
    combined_features = np.concatenate([mfccs_scaled,zero_crossing_rate_scaled, rms_scaled, tempo])
    # combined_features = np.concatenate([mfccs_scaled, chroma_stft_scaled, spectral_centroid_scaled, spectral_bandwidth_scaled, zero_crossing_rate_scaled, rms_scaled, tempo])
    return combined_features

# Removing corrupted files
file_name_to_drop = "jazz.00054.wav"
index_to_drop = metadata[metadata['filename'] == file_name_to_drop].index[0]
metadata = metadata.drop(index_to_drop, axis=0)


from tqdm import tqdm
extracted_features = []
for index_num,row in tqdm(metadata.iterrows()):
    try:
        final_class_labels = row["label"]
        file_name = os.path.join(os.path.abspath(dataset_path), final_class_labels + '/', str(row['filename']))
        data = features_extractor(file_name)
        extracted_features.append([data, final_class_labels])
    except Exception as e:
        print(f"Error: {e}")
        print(f"File: {index_num} is corrupted")
        print(f"File: {row} is corrupted")
        continue
    
print(len(extracted_features))

# Converting extracted_features to Pandas dataframe
extracted_features_df = pd.DataFrame(extracted_features, columns=['feature', 'class'])
extracted_features_df.head()

encoding_map = {"blues": 0, "classical": 1, "country": 2, "disco": 3, "hiphop": 4, "jazz": 5, "pop": 6, "reggae": 7, "rock": 8, "metal": 9}
extracted_features_df['class'] = extracted_features_df['class'].map(encoding_map)
extracted_features_df.head()

# split the dataset
X = np.array(extracted_features_df['feature'].tolist())
y = np.array(extracted_features_df['class'].tolist())

# train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
# model creation
model = Sequential([
    Dense(units=1024, activation="relu"),
    Dense(units=1024, activation="relu"),
    Dense(units=512, activation="relu"),
    Dense(units=512, activation="relu"),
    Dense(units=128, activation="relu"),
    Dense(units=128, activation="relu"),
    Dense(units=32, activation="relu"),
    Dense(units=10, activation="linear")
])

# model compilation
from tensorflow.keras.losses import SparseCategoricalCrossentropy
model.compile(
    loss = SparseCategoricalCrossentropy(from_logits=True),
    optimizer = tensorflow.keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)


# model training
num_epochs = 100
num_batch_size = 32
model.fit(X_train, y_train, batch_size = 32, epochs=100, verbose=1)

model.evaluate(X_test, y_test)

with open('classifier.pkl', 'wb') as file:
    pickle.dump(model, file)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import os
import tensorflow
 


print("This is the starting cwd:")
print(os.getcwd())
dataset_path = "Data/genres_original"
metadata = pd.read_csv("Data/features_30_sec.csv")
print(metadata.head())

def features_extractor(file):
    audio, sample_rate = librosa.load(file, res_type='kaiser_fast')
    # extract mfccs
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled = np.mean(mfccs,axis=1)

    # extract chroma_stfts
    # chroma_stft = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
    # chroma_stft_scaled = np.mean(chroma_stft,axis=1)

    # extract spectral_centroid
    # spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
    # spectral_centroid_scaled = np.mean(spectral_centroid,axis=1)

    # extract spectral_bandwidth
    # spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate)
    # spectral_bandwidth_scaled = np.mean(spectral_bandwidth,axis=1)

    # extract zero_crossing_rate
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=audio)
    zero_crossing_rate_scaled = np.mean(zero_crossing_rate,axis=1)

    # extract rms
    rms = librosa.feature.rms(y=audio)
    rms_scaled = np.mean(rms,axis=1)

    # extract tempo
    onset_env = librosa.onset.onset_strength(y=audio, sr=sample_rate)
    tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sample_rate)
    
    # combine features
    combined_features = np.concatenate([mfccs_scaled,zero_crossing_rate_scaled, rms_scaled, tempo])
    # combined_features = np.concatenate([mfccs_scaled, chroma_stft_scaled, spectral_centroid_scaled, spectral_bandwidth_scaled, zero_crossing_rate_scaled, rms_scaled, tempo])
    return combined_features

# Removing corrupted files
file_name_to_drop = "jazz.00054.wav"
index_to_drop = metadata[metadata['filename'] == file_name_to_drop].index[0]
metadata = metadata.drop(index_to_drop, axis=0)

# Extracting features from dataset
from tqdm import tqdm
extracted_features = []
for index_num,row in tqdm(metadata.iterrows()):
    try:
        final_class_labels = row["label"]
        file_name = os.path.join(os.path.abspath(dataset_path), final_class_labels + '/', str(row['filename']))
        data = features_extractor(file_name)
        extracted_features.append([data, final_class_labels])
    except Exception as e:
        print(f"Error: {e}")
        print(f"File: {index_num} is corrupted")
        print(f"File: {row} is corrupted")
        continue

# Converting extracted_features to Pandas dataframe
extracted_features_df = pd.DataFrame(extracted_features, columns=['feature', 'class'])
print(extracted_features_df.head())

# Encoding labels into int values
encoding_map = {"blues": 0, "classical": 1, "country": 2, "disco": 3, "hiphop": 4, "jazz": 5, "pop": 6, "reggae": 7, "rock": 8, "metal": 9}
extracted_features_df['class'] = extracted_features_df['class'].map(encoding_map)
print(extracted_features_df.head())

# split the dataset
X = np.array(extracted_features_df['feature'].tolist())
y = np.array(extracted_features_df['class'].tolist())

# train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
# model creation
model = Sequential([
    Dense(units=1024, activation="relu"),
    Dense(units=512, activation="relu"),
    Dense(units=512, activation="relu"),
    Dense(units=128, activation="relu"),
    Dense(units=128, activation="relu"),
    Dense(units=32, activation="relu"),
    Dense(units=10, activation="linear")
])

# model compilation
from tensorflow.keras.losses import SparseCategoricalCrossentropy
model.compile(
    loss = SparseCategoricalCrossentropy(from_logits=True),
    optimizer = tensorflow.keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)

# model training
num_epochs = 100
num_batch_size = 32

model.fit(X_train, y_train, batch_size = 32, epochs=100, verbose=1)

model.evaluate(X_test, y_test)

# model.save("classifier.keras")