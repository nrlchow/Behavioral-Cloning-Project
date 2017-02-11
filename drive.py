import argparse
import base64
import json
import cv2
import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO
import scipy
from scipy.misc import imread, imresize

from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array

# Fix error with Keras and TensorFlow
import tensorflow as tf
tf.python.control_flow_ops = tf


sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None


def preprocess(image):
    # Remove top pixels and hood of the car 
    image = image[40:-25,:,:]
    image = imresize(image, (32,32))
    image = adj_brightness(image)
    return image

def adj_brightness(image):
	# Transform image to HSV, Compute random brightness,transform back to RGB 
	# to make it compatibale for simulator input image 
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    random_brightness = 0.25 + np.random.uniform()
    hsv_image[:, :, 2] = hsv_image[:, :, 2] * random_brightness
    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB) 
    

@sio.on('telemetry')
def telemetry(sid, data):
    # The current steering angle of the car
    steering_angle = data["steering_angle"]
    # The current throttle of the car
    throttle = data["throttle"]
    # The current speed of the car
    speed = data["speed"]
    # The current image from the center camera of the car
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    image_array = np.asarray(image)
    
    image_array = preprocess(image_array)
    transformed_image_array = image_array[None, :, :, :]
    # This model currently assumes that the features of the model are just the images. Feel free to change this.
    steering_angle = float(model.predict(transformed_image_array, batch_size=1))
    # The driving model currently just outputs a constant throttle. Feel free to edit this.
    throttle = 0.20      #0.2  
    print(steering_angle, throttle)
    send_control(steering_angle, throttle)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit("steer", data={
    'steering_angle': steering_angle.__str__(),
    'throttle': throttle.__str__()
    }, skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
    help='Path to model definition json. Model weights should be on the same path.')
    args = parser.parse_args()
    with open(args.model, 'r') as jfile:
        # NOTE: if you saved the file by calling json.dump(model.to_json(), ...)
        # then you will have to call:
        #
        #model = model_from_json(json.loads(jfile.read()))
        model = model_from_json(json.load(jfile))
        #
        # instead.
        #model = model_from_json(jfile.read())
        


    model.compile("adam", "mse")
    weights_file = args.model.replace('json', 'h5')
    model.load_weights(weights_file)

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)