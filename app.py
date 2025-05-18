from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS
import os
import json

app = Flask(__name__)
CORS(app)  # Habilita CORS para todas las rutas

# Ruta para verificar que el servidor está activo
@app.route('/')
def home():
    return "¡Servidor activo y funcionando!"

# Ruta para servir el archivo JSON
@app.route('/datos/<filename>')
def serve_json(filename):
    directory = os.path.join(os.getcwd(), "datos")
    try:
        return send_from_directory(directory, filename)
    except FileNotFoundError:
        return jsonify({"error": "Archivo no encontrado"}), 404

# Ruta para buscar productos
@app.route('/buscar', methods=['POST'])
def buscar_producto():
    archivo_json = os.path.join(os.getcwd(), "datos", "product_data_final.json")
    
    # Cargar el archivo JSON
    with open(archivo_json, "r") as f:
        productos = json.load(f)

    # Obtener el nombre del archivo enviado
    if 'imagen' not in request.files:
        return jsonify({"error": "No se encontró el archivo de imagen"}), 400
    
    imagen_subida = request.files['imagen']
    nombre_archivo = imagen_subida.filename
    
    # Buscar el producto correspondiente
    for producto in productos:
        if producto["image_filename"] == nombre_archivo:
            return jsonify({
                "producto": producto["product_url"]
            })
    
    return jsonify({"error": "Producto no encontrado"}), 404

if __name__ == '__main__':
    app.run(debug=True)
