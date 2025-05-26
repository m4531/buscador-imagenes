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
import traceback

# Evitar que TensorFlow intente usar GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

app = Flask(__name__)
CORS(app)

# En Render solo puedes escribir en /tmp
UPLOAD_FOLDER = '/tmp'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Modelo MobileNet - se carga de forma diferida
model = None
vectores_base = None
nombres_imagenes = None
info_productos = None

@app.before_first_request
def inicializar():
    global model, vectores_base, nombres_imagenes, info_productos
    print(">> Cargando modelo MobileNet...")
    base_model = MobileNet(weights='imagenet', include_top=False, pooling='avg')
    model = Model(inputs=base_model.input, outputs=base_model.output)
    print(">> Modelo cargado.")

    with open('datos/vectores_imagenes.pkl', 'rb') as f:
        data = pickle.load(f)
        vectores_base = np.array(data['vectores'])
        nombres_imagenes = data['nombres']
    print(">> Vectores cargados.")

    with open('producto_data.json', 'r') as f:
        metadata = json.load(f)
        info_productos = {item['image_filename']: item for item in metadata}
    print(">> Metadata cargada.")

# Preprocesar imagen para MobileNet
def procesar_imagen(file_path):
    img = image.load_img(file_path, target_size=(224, 224)).convert('RGB')
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    return preprocess_input(arr)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/buscar', methods=['POST'])
def buscar():
    try:
        if 'imagen' not in request.files:
            return 'No se envió ninguna imagen', 400

        archivo = request.files['imagen']
        ruta = os.path.join(app.config['UPLOAD_FOLDER'], archivo.filename)
        archivo.save(ruta)

        print(">> Procesando imagen...")
        img_arr = procesar_imagen(ruta)
        print(">> Imagen procesada. Generando predicción...")
        vector = model.predict(img_arr)
        print(">> Predicción lista.")

        similitudes = cosine_similarity(vector, vectores_base)[0]
        indices_top = np.argsort(similitudes)[::-1][:3]  # puedes ajustar a 5 si todo va bien

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
        print(f"[ERROR] {e}")
        traceback.print_exc()
        return f'Error al procesar la imagen: {e}', 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
