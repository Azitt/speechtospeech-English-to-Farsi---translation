import random
import os
from flask import Flask, request, send_file, jsonify
from sts import STS_model
import json
from scipy.io.wavfile import write
from os import path
from pydub import AudioSegment
import uuid

# instantiate flask app
app = Flask(__name__)
STS = STS_model()

@app.route("/predict", methods=["POST"])
def predict():

    # get file from POST request and save it
    audio_file = request.files["file"]
    file_name = uuid.uuid4().hex
    file_path = f'static/audio/inputs/{file_name}.wav'
    print(audio_file)
    print("start of translator..")
    # files
    # sample_width=2,frame_rate=22050,channels=1
    wav = AudioSegment.from_file(audio_file)
    wav.export(file_path, format="wav")
    # instantiate keyword spotting service singleton and get prediction
    # STS = STS_model()
    model_output = STS.predict(file_path)

    return jsonify(
        st_out=model_output[0],
		translate_out=model_output[1],
		output_path=model_output[2]
    )


if __name__ == "__main__":
    #app.run(debug=True)
    print("flask app start")
    app.run(host="0.0.0.0", port=5000, debug=True)
