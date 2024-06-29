import io
import os

from flask import Flask, jsonify, render_template, request
from PIL import Image

from prediction_service import prediction  # Assuming this module exists

# Initialize Flask app
app = Flask(__name__)
webapp_root = "webapp"
app.template_folder = os.path.join(webapp_root, "templates")
app.static_folder = os.path.join(webapp_root, "static")


@app.route("/", methods=["GET"])
def index():
    # Render the main page
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def upload():
    if request.method == "POST":
        # Validate if file is in the request
        if "file" not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files["file"]

        # Validate if a file is selected
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        # Validate if file is an image
        try:
            image_data = file.read()
            image = Image.open(io.BytesIO(image_data))
            image.verify()
        except Exception as e:
            return jsonify({"error": "Invalid image file"}), 400

        # Get predictions using the form_response function
        result = prediction.form_response(image)

        # Return the prediction result as a JSON response
        return jsonify(result)

    return None


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
