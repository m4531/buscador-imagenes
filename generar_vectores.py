# generar_vectores.py
import os
import pickle
import numpy as np
import json
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

base_model = MobileNet(weights='imagenet', include_top=False, pooling='avg')
model = Model(inputs=base_model.input, outputs=base_model.output)

with open('producto_data.json', 'r') as f:
    metadata = json.load(f)

imagenes_validas = {item['image_filename']: item for item in metadata}
vectores = []
nombres = []
carpeta = 'imagenes_catalogo'

for nombre_img in imagenes_validas:
    path = os.path.join(carpeta, nombre_img)
    if os.path.exists(path):
        img = image.load_img(path, target_size=(224, 224))
        arr = image.img_to_array(img)
        arr = np.expand_dims(arr, axis=0)
        arr = preprocess_input(arr)
        vector = model.predict(arr)[0]
        vectores.append(vector)
        nombres.append(nombre_img)
        print(f"Procesado: {nombre_img}")
    else:
        print(f"No encontrado: {nombre_img}")

with open("datos/vectores_imagenes.pkl", "wb") as f:
    pickle.dump({"vectores": vectores, "nombres": nombres}, f)

print("Vectores guardados en datos/vectores_imagenes.pkl")
