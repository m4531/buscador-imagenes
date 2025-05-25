
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import numpy as np
import pickle
import os
import json
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'imagenes_catalogo'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

base_model = MobileNet(weights='imagenet', include_top=False, pooling='avg')
model = Model(inputs=base_model.input, outputs=base_model.output)

with open('datos/vectores_imagenes.pkl', 'rb') as f:
    data = pickle.load(f)
    vectores_base = np.array(data['vectores'])
    nombres_imagenes = data['nombres']

with open('producto_data.json', 'r') as f:
    metadata = json.load(f)
    info_productos = {item['image_filename']: item for item in metadata}

def procesar_imagen(file_path):
    img = image.load_img(file_path, target_size=(224, 224))
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    return preprocess_input(arr)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/buscar', methods=['POST'])
def buscar():
    if 'imagen' not in request.files:
        return 'No se envió ninguna imagen', 400
    archivo = request.files['imagen']
    ruta = os.path.join(app.config['UPLOAD_FOLDER'], archivo.filename)
    archivo.save(ruta)
    try:
        img_arr = procesar_imagen(ruta)
        vector = model.predict(img_arr)
        similitudes = cosine_similarity(vector, vectores_base)[0]
        indices_top = np.argsort(similitudes)[::-1][:5]
        resultados = []
        for i in indices_top:
            nombre_img = nombres_imagenes[i]
            producto = info_productos.get(nombre_img, {})
            resultados.append({
                "nombre": producto.get("nombre", "Producto desconocido"),
                "imagen_local": nombre_img,
                "imagen_url": producto.get("image_url", ""),
                "producto_url": producto.get("product_url", "#")
            })
        return jsonify(resultados)
    except Exception as e:
        return f'Error al procesar la imagen: {e}', 500

if __name__ == '__main__':
    app.run(debug=True)
