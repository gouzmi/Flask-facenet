import flask
from uuid import uuid4
from flask import Flask, request, render_template, send_from_directory
from sklearn.externals import joblib
import numpy as np
from scipy import misc
import cv2
from werkzeug.utils import secure_filename
import os
from keras.applications import VGG16
from keras.preprocessing import image
from keras.models import Sequential
from keras.applications.vgg16 import preprocess_input, decode_predictions
import pandas as pd
import tensorflow as tf

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route("/")
def index():
    return render_template("upload.html")

@app.route("/upload", methods=["POST"])
def upload():
    target = os.path.join(APP_ROOT, 'static/')
    print(target)
    if not os.path.isdir(target):
            os.mkdir(target)
    else:
        print("Couldn't create upload directory: {}".format(target))
    print(request.files.getlist("file"))
    for upload in request.files.getlist("file"):
        print(upload)
        print("{} is the file name".format(upload.filename))
        filename = upload.filename
        destination = "/".join([target, filename])
        print ("Accept incoming file:", filename)
        print ("Save it to:", destination)
        upload.save(destination)

    return render_template("display.html", image_name=filename)
    #return send_from_directory("static", filename)

#     return send_from_directory("static", filename)

#     #Faire les tranfos images

#     #Aller chercher dans le csv les images correspondantes

#     #Return la liste

if __name__ == '__main__':	
	# load ml model

	# start api
	app.run(debug=True)
