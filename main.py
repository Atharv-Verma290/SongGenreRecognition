from flask import Flask, render_template, request, redirect, url_for
import tensorflow
from tensorflow import keras
import numpy as np
import librosa

app = Flask(__name__)

@app.route('/')
def index():
  return render_template('homePage.html')

def feature_extractor(file):
  audio, sample_rate = librosa.load(file, res_type='kaiser_best', duration=30)

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



def decoder(value):
  decoded_value = None
  match value:
    case 0: decoded_value = "blues"
    case 1: decoded_value = "classical"
    case 2: decoded_value = "country"
    case 3: decoded_value = "disco"
    case 4: decoded_value = "hiphop"
    case 5: decoded_value = "jazz"
    case 6: decoded_value = "pop"
    case 7: decoded_value = "reggae"
    case 8: decoded_value = "rock"
    case 9: decoded_value = "metal"

  return decoded_value


def allowed_file(filename):
    return filename.endswith(".wav")


@app.route('/test1', methods = ['POST', 'GET'])
def handle_upload():
  if request.method == "POST":
    file = request.files["file"]
    if file and allowed_file(file.filename):
      #do features extraction
      data = feature_extractor(file)
      data = np.reshape(data, (1, -1))  # Reshape the data
      print(data)
      print(data.shape)
      genre_classifier = keras.models.load_model("classifier.keras", compile=False)
      predicted_class = np.argmax(genre_classifier.predict(data))
      predicted_genre = decoder(predicted_class)

      return f"Your predicted genre is {predicted_genre}."
    return "<h1>Your file is not correct.</h1>"
  else:
    return render_template("test1.html")



if __name__ == '__main__':
  app.run(debug=True)

