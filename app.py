from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import zipfile
import os
import tempfile
import langchain
from PIL import Image
from models.knowledge_base import crack_knowledge_base 

app = Flask(__name__)

def load_models():
    from models.deepcrack_model import DeepCrackModel
    from models.roadnet_model import RoadNetModel
    from models.base_model import BaseModel
    from models.deepcrack_networks import DeepCrackNetworks
    from models.networks import Networks

    deepcrack_model = DeepCrackModel().build()
    roadnet_model = RoadNetModel().build()
    base_model = BaseModel().build()
    deepcrack_networks = DeepCrackNetworks().build()
    networks_model = Networks().build()

    return deepcrack_model, roadnet_model, base_model, deepcrack_networks, networks_model

def generate_summary_and_maintenance(detection_results):
    crack_type = detection_results['type']
    crack_properties = crack_knowledge_base.get(crack_type.lower(), {
        "description": "Unknown crack type.",
        "maintenance": "No maintenance guidelines available."
    })

    input_text = (
        f"Detected {detection_results['num_cracks']} cracks of type '{crack_type}'. "
        f"Description: {crack_properties['description']}. "
        f"Severity levels: {detection_results['severity_levels']}. "
        f"Locations: {detection_results['locations']}. "
        f"Recommended maintenance: {crack_properties['maintenance']}."
    )

    summary = langchain.llm_generate(input_text)
    return summary

def load_and_preprocess_image(img_path):
    img = Image.open(img_path).convert('L')
    img = img.resize((512, 512))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=(0, -1))
    return img_array

def process_image(img_path):
    deepcrack_model, roadnet_model, base_model, deepcrack_networks, networks_model = load_models()
    img_array = load_and_preprocess_image(img_path)
    

    deepcrack_result = deepcrack_model.predict(img_array)
    roadnet_result = roadnet_model.predict(img_array)
    base_model_result = base_model.predict(img_array)
    deepcrack_network_result = deepcrack_networks.predict(img_array)
    networks_result = networks_model.predict(img_array)
    
    detection_results = {
        'num_cracks': int(deepcrack_result.sum() + roadnet_result.sum() +
                          base_model_result.sum() + deepcrack_network_result.sum() +
                          networks_result.sum()),
        'type': "shear",  # Placeholder; add logic for crack type detection
        'severity_levels': "Moderate", 
        'locations': "Center, Top-Left"
    }
    
    return detection_results


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
    
    with tempfile.TemporaryDirectory() as tmpdirname:
        if file.filename.endswith('.zip'):
            with zipfile.ZipFile(file, 'r') as zip_ref:
                zip_ref.extractall(tmpdirname)
            
            summaries = []
            for img_name in os.listdir(tmpdirname):
                img_path = os.path.join(tmpdirname, img_name)
                if img_path.endswith(('png', 'jpg', 'jpeg')):
                    detection_results = process_image(img_path)
                    summary = generate_summary_and_maintenance(detection_results)
                    summaries.append({"image": img_name, "summary": summary, "detection_results": detection_results})
            
            return render_template('result.html', summaries=summaries)

        else:
            img_path = os.path.join(tmpdirname, file.filename)
            file.save(img_path)
            detection_results = process_image(img_path)
            summary = generate_summary_and_maintenance(detection_results)
            return render_template('result.html', summaries=[{"image": file.filename, "summary": summary, "detection_results": detection_results}])

if __name__ == '__main__':
    app.run(debug=True)
