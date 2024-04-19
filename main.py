from flask import Flask, render_template, request
import tensorflow
from tensorflow import keras
import numpy as np
import librosa

app = Flask(__name__)

@app.route('/')
def index():
  return render_template('homePage.html')

def extract_features(file):
  audio, sample_rate = librosa.load(file, res_type='kaiser_fast')
  # extract mfccs
  mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
  mfccs_scaled = np.mean(mfccs,axis=1)
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

  

  return combined_features

def classification():
  pass

@app.route('/process', methods = ['POST', 'GET'])
def handle_upload():
  raw_file = request.files['file']
  if raw_file.filename != '':
    if raw_file.filename.lower().endswith('.wav'):
      return "File is a valid .wav file."
    else:
      return "File is invalid."
  else:
    return "No file uploaded."
  return render_template('process.html')



if __name__ == '__main__':
  app.run(debug=True)

