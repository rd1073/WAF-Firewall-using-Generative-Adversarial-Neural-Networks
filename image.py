from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import easyocr

app = Flask(__name__)
CORS(app)

# Initialize EasyOCR reader once
reader = easyocr.Reader(['en'])  # Add other languages like ['en', 'hi'] if needed

@app.route('/upload', methods=['POST'])
def upload_images():
    files = request.files.getlist('images')
    if len(files) > 20:
        return jsonify({"error": "You can upload up to 20 images only."}), 400

    extracted_texts = []

    for i, file in enumerate(files):
        try:
            image = Image.open(file.stream).convert("RGB")
            image_np = np.array(image)

            result = reader.readtext(image_np)
            text = "\n".join([line[1] for line in result])

            extracted_texts.append({
                "image_number": i + 1,
                "text": text.strip()
            })
        except Exception as e:
            extracted_texts.append({
                "image_number": i + 1,
                "text": f"Error reading image: {str(e)}"
            })

    return jsonify(extracted_texts)

if __name__ == '__main__':
    app.run(debug=True)
