#==============================================================
                        #imports
#==============================================================
import tensorflow as tf
import cv2
import numpy as np
import io
from flask import Flask, flash, request, render_template, redirect
from load_model import *

UPLOAD_FOLDER = 'static/uploads/'

app = Flask(__name__)
app.secret_key = "suprrrrrrr secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# @tf.keras.utils.register_keras_serializable()
def custom_standardization(s):
    s = tf.strings.lower(s)
    s = tf.strings.regex_replace(s, f'[{re.escape(string.punctuation)}]', '')
    s = tf.strings.join(['[START]', s, '[END]'], separator=' ')
    return s

def predictions(image):
    image = tf.convert_to_tensor(image, dtype=tf.int32)
    image = load_image_obj(image)

    comments = []
    for t in [0, 0.5, 1]:
        comment = model.simple_gen(image, t)
        comments.append(comment)
    return comments


#==============================================================
                        #routes
#==============================================================

@app.route('/', methods=['GET'])
def main():
    return render_template("upload.html")


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method=="GET":
        return redirect(request.url)
    if "image" not in request.files:
        return "image not found"
    
    photo = request.files['image']

    if photo.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    
    in_memory_file = io.BytesIO()
    photo.save(in_memory_file)
    data = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8)
    color_image_flag = 1
    img = cv2.imdecode(data, color_image_flag)
    comments = predictions(img)

    return render_template("result.html", comments=comments)

if __name__=="__main__":
    model = build()

    app.run(host="0.0.0.0", port=5000)

print("--Terminated--")