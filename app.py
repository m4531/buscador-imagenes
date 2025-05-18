from flask import Flask, request, jsonify
from PIL import Image
import imagehash
import os

app = Flask(__name__)
BASE_DIR = "imagenes_catalogo"

def calcular_hash(imagen_path):
    with Image.open(imagen_path) as img:
        return imagehash.phash(img)

@app.route('/comparar', methods=['POST'])
def comparar():
    if 'imagen' not in request.files:
        return jsonify({'error': 'No se subió ninguna imagen'}), 400

    imagen_subida = request.files['imagen']
    hash_subido = calcular_hash(imagen_subida)

    resultados = []

    for filename in os.listdir(BASE_DIR):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            path_img = os.path.join(BASE_DIR, filename)
            hash_img = calcular_hash(path_img)
            diferencia = hash_subido - hash_img
            resultados.append({'imagen': filename, 'diferencia': int(diferencia)})

    resultados.sort(key=lambda x: x['diferencia'])

    return jsonify({'resultados': resultados[:5]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
