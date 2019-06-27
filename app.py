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
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input, decode_predictions
import pandas as pd
import tensorflow as tf
import FaceNet as fn

app = Flask(__name__)

global graph
model_path = 'facenet_keras.h5'
modele = load_model(model_path)
modele.load_weights('facenet_keras_weights.h5')
graph = tf.get_default_graph()


APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route("/")
def index():
    return render_template("upload.html")

@app.route("/upload", methods=['POST'])
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
        destination = ''.join([target, filename])
        print ("Accept incoming file:", filename)
        print ("Save it to:", destination)
        upload.save(destination)
        #faire le bail
        values = pd.read_csv('features.csv')
        photo = destination
        
        try:
            values['result'] = fn.euclidean_distances(values.iloc[:,1:],fn.facenet(photo,modele,10,graph))
            imgs = (values.sort_values(by='result').iloc[:5,0]).tolist()
            scores = (values.sort_values(by='result').iloc[:5,-1]).tolist()
            scores = [ round(elem, 3) for elem in scores ]
        except:
            msg="Aucun visage n'a été trouvé"
            return render_template("upload.html",msg=msg)
    return render_template("display.html", image_name=filename, imgs = imgs, len = len(imgs),scores = scores)
    #return send_from_directory("static", filename)

if __name__ == '__main__':	
	# load ml model
    
	# start api http://127.0.0.1:5000/
    app.run(debug=True)