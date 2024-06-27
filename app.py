import os
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from prediction_service import prediction  # Assuming this module exists
from PIL import Image
import io

# Project Structure
app = Flask(__name__)
webapp_root = "webapp"
app.template_folder = os.path.join(webapp_root, "templates")
app.static_folder = os.path.join(webapp_root, "static")

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET", "POST"])
def handle_form_or_json():
    """Handle both form submissions and JSON payloads."""
    if request.method == "GET":
        return render_template("index.html")
    try:
        if request.files:
            image = request.files['image']
            if image and allowed_file(image.filename):
                # Read the image file directly into memory
                image_bytes = image.read()
                image_pil = Image.open(io.BytesIO(image_bytes))
                print(image_pil)
                response = prediction.form_response(image_pil)
                print(response)
                return render_template("index.html", response=response)
            else:
                return render_template("error.html", error="Invalid file type. Please upload an image file.")

        elif request.json:
            json_data = request.json
            response = prediction.api_response(json_data)
            return jsonify(response)
        
        else:
            return render_template("error.html", error="No file or JSON data received.")

    except Exception as e:
        error = str(e)
        return render_template("error.html", error=error)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)