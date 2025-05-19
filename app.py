from flask import Flask, request, jsonify
from flask_cors import CORS
import imagehash
from PIL import Image
import json
import os

app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def home():
    return "Servidor activo y funcionando"

@app.route("/buscar", methods=["POST"])
def buscar():
    if 'imagen' not in request.files:
        return jsonify({"error": "No se encontró el archivo de imagen"}), 400

    imagen_subida = request.files['imagen']
    img = Image.open(imagen_subida.stream)
    hash_subido = imagehash.average_hash(img)

    # Cargar los hashes de productos
    with open("datos/hashes_productos.json", "r") as f:
        productos = json.load(f)

    resultados = []

    for producto in productos:
        hash_producto = imagehash.hex_to_hash(producto["image_hash"])
        distancia = hash_subido - hash_producto  # Distancia Hamming
        resultados.append((distancia, producto))

    resultados.sort(key=lambda x: x[0])

    top_resultados = [r[1] for r in resultados[:5]]

    return jsonify(top_resultados)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
