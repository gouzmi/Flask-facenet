import flask
from flask import Flask, request, render_template
from sklearn.externals import joblib
import numpy as np
from scipy import misc
import cv2	
from keras.applications import VGG16
from keras.preprocessing import image
from keras.models import Sequential
from keras.applications.vgg16 import preprocess_input, decode_predictions
import pandas as pd
import tensorflow as tf
global graph, model
graph = tf.get_default_graph()

app = Flask(__name__)

@app.route("/")
@app.route("/index")
def index():
    return flask.render_template('index.html')


@app.route('/predict', methods=['POST'])
def make_prediction():
	if request.method=='POST':

		# get uploaded image file if it exists
		file = request.files['image']
		if not file: return render_template('index.html', label="No file")
		
		img = misc.imread(file)
		img_data = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
		# img = tf.image.resize_images(img,[224,224])
		#img = image.load_img(file, target_size=(224, 224))
		#img_data = image.img_to_array(img)
		img_data = np.expand_dims(img_data, axis=0)
		img_data = preprocess_input(img_data)	
		with graph.as_default():
    			preds = VGG16.predict(img_data)
		label = decode_predictions(preds,top=1)[0][0][1] + (' ') +str(decode_predictions(preds,top=1)[0][0][2])

		return render_template('index.html', label=label)


if __name__ == '__main__':	
	# load ml model
	VGG16 = VGG16(weights='imagenet')
	# start api
	app.run(host='0.0.0.0', port=8000, debug=True)
