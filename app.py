import io
import os
from PIL import Image
from flask import Flask, jsonify, render_template, request, g
from prediction_service import prediction

app = Flask(__name__)
webapp_root = "webapp"
params_path = "params.yaml"
app.template_folder = os.path.join(webapp_root, "templates")
app.static_folder = os.path.join(webapp_root, "static")

def get_model():
    if 'model' not in g:
        g.model, g.model_config = prediction.load_model(params_path)
    return g.model, g.model_config

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        image_data = file.read()
        image = Image.open(io.BytesIO(image_data))
        image.verify()
    except Exception:
        return jsonify({"error": "Invalid image file"}), 400

    # Assuming the model and prediction logic is implemented in these functions
    model, model_config = get_model()
    
    # Get predictions using the form_response function
    result = prediction.response(image,model,model_config)
    
    # Return the prediction result as a JSON response
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True) # for localhost
    # app.run(host="0.0.0.0", port=8080, debug=True) # for AWS 
    # app.run(host="0.0.0.0", port=80, debug=True) # for Azure 