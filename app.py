import os
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from prediction_service import prediction  # Assuming this module exists
from PIL import Image
import io

# Initialize Flask app
app = Flask(__name__)
webapp_root = "webapp"
app.template_folder = os.path.join(webapp_root, "templates")
app.static_folder = os.path.join(webapp_root, "static")

@app.route('/', methods=['GET'])
def index():
    # Render the main page
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        # Get the file from the POST request
        file = request.files['file']
        
        # Open the image file and read its contents
        image_data = file.read()
        
        # Create an Image object from the image data
        image = Image.open(io.BytesIO(image_data))
        
        # Get predictions using the form_response function
        result = prediction.form_response(image)
        
        # Return the prediction result as a JSON response
        return jsonify(result)
    
    return None

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
