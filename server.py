from typing import Tuple, List, Text, Dict, Any, Iterator
import sys
import cv2
import numpy as np
import time
from flask import Flask, request, json, send_file

import keras.backend as  K
from keras.models import model_from_json
import skimage.io as io
from model_unet import create_unet

folder = "/data/yosuke/"

app = Flask(__name__, static_url_path='')

with K.tf.device('/cpu:0'):
    model = create_unet((512, 512, 3), 1, 64)
    model.load_weights(folder+"2017-04-28-11-11-28_fil64_adam_lr0.0001_glorot_uniform_dice_coef_weights.epoch0075-val_loss-0.79-val_dice_coef0.79.hdf5")

@app.route('/')
def root():
    return app.send_static_file('index.html')

@app.route('/post', methods=['POST'])
def upload_file():
    if not request.method == 'POST':
        print("post it")
        return app.response_class(status=400)

    files = request.files.getlist("files")

    _files = []
    for (i, file) in enumerate(files):
        name = ("/tmp/img%d.img" % i)
        file.save(name)
        _files.append(name)

    if len(_files) == 0:
        print("no file")
        return app.response_class(status=400)
    
    filename = _files[0]

    print("processing...", _files)
    start = time.time()

    img = io.imread(filename)
    img = cv2.resize(img, (512, 512))
    img = np.expand_dims(img, axis=0)

    with K.tf.device('/cpu:0'):
        output = model.predict(img)
    filename += ".png"
    io.imsave(filename, output[0])
    
    elapsed = time.time() - start
    print(elapsed, "sec, ", filename)

    res = send_file(filename, mimetype='image/png')

    h = res.headers
    h['Access-Control-Allow-Origin'] = "*"
    h['Access-Control-Allow-Methods'] = "POST"
    h['Access-Control-Max-Age'] = "21600"

    return res





