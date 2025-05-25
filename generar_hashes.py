from PIL import Image
import imagehash
import os
import json

# Cambia esta ruta a donde tengas tus imágenes
IMAGES_FOLDER = IMAGES_FOLDER = r"C:\Users\Jamie\Downloads\buscador-imagenes\imagenes_catalogo"
# Archivo donde se guardarán los hashes
OUTPUT_JSON = "datos/hashes_productos.json"

hashes = []

for filename in os.listdir(IMAGES_FOLDER):
    if filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
        path = os.path.join(IMAGES_FOLDER, filename)
        img = Image.open(path)
        hash_str = str(imagehash.average_hash(img))
        hashes.append({
            "image_filename": filename,
            "image_hash": hash_str,
            "product_url": f"https://tusitio.com/producto/{filename.split('.')[0]}",
            "image_url": f"https://tusitio.com/wp-content/uploads/{filename}",
            "nombre": filename.split('.')[0].replace('-', ' ').title()
        })

with open(OUTPUT_JSON, "w") as f:
    json.dump(hashes, f, indent=2)

print(f"Archivo JSON generado en {OUTPUT_JSON}")