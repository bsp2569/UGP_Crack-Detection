from flask import Flask, render_template, request, redirect, url_for, jsonify
import numpy as np
import zipfile
import os
import tempfile
import langchain
from PIL import Image

app = Flask(__name__)


# Function to generate a summary using LangChain
def generate_summary(detection_results):
    input_text = f"Detected {detection_results['num_cracks']} cracks. Severity levels: {detection_results['severity_levels']}. Locations: {detection_results['locations']}."
    return langchain.llm_generate(input_text)

# Function to load and preprocess an image using Pillow
def load_and_preprocess_image(img_path):
    img = Image.open(img_path).convert('L')  # Convert to grayscale
    img = img.resize((512, 512))             # Resize to model input size
    img_array = np.array(img) / 255.0        # Normalize pixel values to [0, 1]
    img_array = np.expand_dims(img_array, axis=(0, -1))  # Add batch and channel dimensions
    return img_array


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    # Create a temporary directory to store images if ZIP file is uploaded
    with tempfile.TemporaryDirectory() as tmpdirname:
        if file.filename.endswith('.zip'):
            # Extract ZIP file
            with zipfile.ZipFile(file, 'r') as zip_ref:
                zip_ref.extractall(tmpdirname)
            
            # Process each image in the extracted folder
            summaries = []
            for img_name in os.listdir(tmpdirname):
                img_path = os.path.join(tmpdirname, img_name)
                if img_path.endswith(('png', 'jpg', 'jpeg')):
                    detection_results = process_image(img_path)
                    summary = generate_summary(detection_results)
                    summaries.append({"image": img_name, "summary": summary, "detection_results": detection_results})
            
            return render_template('result.html', summaries=summaries)

        else:
            # Process a single image upload
            img_path = os.path.join(tmpdirname, file.filename)
            file.save(img_path)
            detection_results = process_image(img_path)
            summary = generate_summary(detection_results)
            return render_template('result.html', summaries=[{"image": file.filename, "summary": summary, "detection_results": detection_results}])

if __name__ == '__main__':
    app.run(debug=True)
