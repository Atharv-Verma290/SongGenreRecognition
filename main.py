from flask import Flask, render_template
import tensorflow
from tensorflow import keras

app = Flask(__name__)

@app.route('/')
def index():
  return render_template('home.html')

if __name__ == '__main__':
  app.run(debug=True)

