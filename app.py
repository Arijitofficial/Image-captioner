from flask import Flask, flash, request, render_template, redirect
from werkzeug.utils import secure_filename
from gradio_client import Client
import os

client = Client("https://arijit-hazra-my-image-captioner.hf.space/")
UPLOAD_FOLDER = 'static/uploads/'

app = Flask(__name__)
app.secret_key = "suprrrrrrr secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

host = "0.0.0.0"
port = 5000
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

# =======================================================================================
#                                          utils
# =======================================================================================


def predict_api(image_path, delete = False):
    if image_path=="":
        image_path="https://www.surfertoday.com/images/stories/surfingdog.jpg"
    
    result = client.predict(
            image_path,	# str representing filepath or URL to image in 'image' Image component
            api_name="/predict"
    )

    if delete:
        os.remove(image_path)

    return render_template("result.html", comments=result)


def allowed_file(filename):
    return filename.split('.')[1] in ALLOWED_EXTENSIONS

def upload_file(request, filename="image"):
    if request.method == 'POST':
        if filename not in request.files:
            flash('No file part')
            return False
        file = request.files[filename]

        if file.filename == '':
            flash('No selected file')
            return False
        if file and allowed_file(file.filename):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
            file.save(filepath)
            return filepath
    return False


# =======================================================================================
#                                          routes
# =======================================================================================

# home
@app.route('/', methods=['GET'])
def main():
    return render_template("upload.html")


# prediction result
@app.route('/predict', methods=['POST', 'GET'])
def predict():
    
    if request.method=="GET":
        return predict_api("")
    if "image" not in request.files:
        return predict_api("")
    
    photo = request.files['image']

    if photo.filename == '':
        flash('No image selected for processing')
        return predict_api("")
    
    filepath = upload_file(request)
    if filepath:
        return predict_api(filepath, delete=True)
    print(filepath)
    return predict_api("")


if __name__=="__main__":
    app.run(host=host, port=port)

print("--Terminated--")