from flask import Flask, render_template, request, send_from_directory
import cv2
import numpy as np
import os
from sklearn.cluster import KMeans

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Route to serve uploaded images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Function to extract top colors using KMeans
def get_dominant_colors(image_path, k=10):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (200, 200))  # Resize for speed

    flat_img = img.reshape((-1, 3))

    kmeans = KMeans(n_clusters=k, n_init='auto')
    kmeans.fit(flat_img)

    colors = kmeans.cluster_centers_.astype(int)
    hex_colors = ['#%02x%02x%02x' % tuple(color) for color in colors]

    return hex_colors

# Home route
@app.route('/', methods=['GET', 'POST'])
def index():
    colors = []
    filename = None

    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            colors = get_dominant_colors(filepath)

    return render_template('index.html', colors=colors, filename=filename)

# Run the app
if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)
