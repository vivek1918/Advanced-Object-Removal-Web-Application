from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
from io import BytesIO
import base64
from PIL import Image

app = Flask(__name__)

# Route to handle the uploaded image
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the image from the form
        file = request.files['file']
        img = Image.open(file.stream)
        
        # Convert image to numpy array for OpenCV
        img = np.array(img)
        
        # Convert to BGR for OpenCV processing
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Save the image to the server
        cv2.imwrite('uploaded_image.jpg', img)
        return render_template('index.html')

    return render_template('index.html')

# Route to handle the inpainting of the drawn region
@app.route('/remove-object', methods=['POST'])
def remove_object():
    data = request.get_json()
    img_data = data['image'].split(',')[1]  # Strip base64 prefix

    # Decode the base64 image
    nparr = np.frombuffer(base64.b64decode(img_data), np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Create a blank mask
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # Process the drawn paths and create a mask
    for path in data['paths']:
        points = np.array(path, dtype=np.int32)
        cv2.fillPoly(mask, [points], 255)  # Fill the path with white in the mask

    # Inpaint the image using the mask
    result = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    # Encode the result back to base64 to send to the frontend
    _, buffer = cv2.imencode('.jpg', result)
    result_data = base64.b64encode(buffer).decode('utf-8')

    return jsonify({'message': 'Object Removed Successfully!', 'result_image': result_data})


if __name__ == '__main__':
    app.run(debug=True)
