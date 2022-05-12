# import necessities
from tensorflow.keras.models import load_model
from mtcnn_cv2 import MTCNN
from io import BytesIO
from PIL import Image

from flask import Response
from flask import Flask, request

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import normalize

from operator import attrgetter
from pathlib import Path
from glob import glob

import numpy as np
import requests
import base64
import json
import time
import cv2
import os

import sys
sys.path.append("./yolo-v4-tf.keras")
from models import Yolov4


# init webserver: flask app
app = Flask(__name__)
app.config["DEBUG"] = True

# prepare consts
WEIGHT_PATH = 'yolo/yolov4.weights'
CLASSES_PATH = 'yolo-v4-tf.keras/class_names/coco_classes.txt'
DIRNAME = 'tmp'
FILE = 'tmpfile'  # TODO: avoid collisions in tmpfile name, generate name on-the-fly
SUFFIX = '.jpeg'  # TODO: is jpeg encoding faster? verify
FNAME = f"{DIRNAME}/{FILE}{SUFFIX}"
MODEL = Yolov4(weight_path=WEIGHT_PATH, class_name_path=CLASSES_PATH)

# detect route:
#   action: 
#   input: image: base64 string, blob
#   output: Resp([200]): detections: bbox, class, probability 
@app.route('/enroll', methods=['POST'])
def enroll():
    
    req_image = None
    img_bytes = None
    faces = None
    
    # check for image
    if "image" not in request.form and "image" not in request.files:
        return "image not found in request."
        
    elif "image" in request.files and request.files["image"].filename == "": 
        return "Bad request. No image found in upload."
    
    # ingest image
    elif "image" in request.form:    
        try:
            req_image = request.form['image']  # handle image data in POST body
            
            # jpeg base64
            if "data:image/jpeg;base64," in req_image:
                base_string = req_image.replace("data:image/jpeg;base64,", "")  # TODO: verify non-empty body
                img_bytes = base64.b64decode(base_string)
            
            # png base64
            elif "data:image/png;base64," in req_image:
                base_string = req_image.replace("data:image/png;base64,", "")
                img_bytes = base64.b64decode(base_string)
                
            # image url
            elif "https://" in req_image:
                response = requests.get(req_image) # TODO: verify no errors
                img_bytes = response.content
            
        except Exception as e:
            print(e)
            return "image ingestion failed"
            
    elif "image" in request.files and request.files['image'].filename != "":  
        img_bytes = request.files['image'].read()  # handle image data in upload
        
    try:
        # write img_bytes to file
        with open(FNAME, 'wb') as fname:
            fname.write(img_bytes)
        
        predictions = MODEL.predict(FNAME, plot_img=False)
        return Response(predictions=predictions.to_json()), status=200)
        
    except Exception as e:
        print(e)
        return "image decoding failed"

# preprocess image

# run detection

# return detections
