# generar_vectores.py
import os
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import pickle

# Ruta a la carpeta de imágenes
IMAGENES_DIR = 'imagenes_catalogo'

# Cargar modelo MobileNet sin la última capa (para embeddings)
base_model = MobileNet(weights='imagenet', include_top=False, pooling='avg')
model = Model(inputs=base_model.input, outputs=base_model.output)

def cargar_y_procesar_imagen(ruta):
    img = image.load_img(ruta, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

vectores = []
nombres_archivos = []

for nombre in os.listdir(IMAGENES_DIR):
    ruta_img = os.path.join(IMAGENES_DIR, nombre)
    try:
        img_procesada = cargar_y_procesar_imagen(ruta_img)
        vector = model.predict(img_procesada)[0]
        vectores.append(vector)
        nombres_archivos.append(nombre)
        print(f"Procesada: {nombre}")
    except Exception as e:
        print(f"Error con {nombre}: {e}")

# Guardar vectores y nombres de archivo
with open('vectores_imagenes.pkl', 'wb') as f:
    pickle.dump({'vectores': vectores, 'nombres': nombres_archivos}, f)

print("Listo: vectores guardados en vectores_imagenes.pkl")
