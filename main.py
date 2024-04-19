from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def index():
  return render_template('homePage.html')

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

