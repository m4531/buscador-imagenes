from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS
import os
import json

app = Flask(__name__)
CORS(app)  # Habilitar CORS para todas las rutas

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

# Ruta para buscar productos parecidos
@app.route('/buscar', methods=['POST'])
def buscar_producto():
    archivo_json = os.path.join(os.getcwd(), "datos", "product_data_final.json")
    
    # Cargar el archivo JSON
    with open(archivo_json, "r") as f:
        productos = json.load(f)

    if 'imagen' not in request.files:
        return jsonify({"error": "No se encontró el archivo de imagen"}), 400
    
    imagen_subida = request.files['imagen']
    nombre_archivo = imagen_subida.filename.lower()

    # Aquí deberías implementar la lógica de búsqueda por similitud
    # Por ahora, simulamos que buscamos y devolvemos hasta 3 productos que coincidan parcialmente por nombre (ejemplo simple)

    resultados = []
    for producto in productos:
        if nombre_archivo in producto["image_filename"].lower() or producto["image_filename"].lower() in nombre_archivo:
            resultados.append({
                "image_url": producto.get("image_url", ""),  # Añade la URL completa de la imagen aquí
                "nombre": producto.get("nombre", producto["image_filename"]),
                "product_url": producto["product_url"]
            })
            if len(resultados) >= 3:  # Limitar a 3 resultados
                break

    if resultados:
        return jsonify({"productos": resultados})
    else:
        return jsonify({"error": "No se encontraron productos parecidos"}), 404

if __name__ == '__main__':
    app.run()
