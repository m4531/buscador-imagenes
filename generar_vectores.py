import os
import numpy as np
import pickle
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

# Ruta donde están tus imágenes
RUTA_IMAGENES = "imagenes_catalogo"

# Carga modelo MobileNet sin capa superior
base_model = MobileNet(weights='imagenet', include_top=False, pooling='avg')
model = Model(inputs=base_model.input, outputs=base_model.output)

# Procesa una imagen y retorna el vector
def extraer_vector(imagen_path):
    img = image.load_img(imagen_path, target_size=(224, 224)).convert('RGB')
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    vector = model.predict(arr)
    return vector.flatten().tolist()  # convertimos a lista nativa

vectores = []
nombres = []

# Itera sobre imágenes en el directorio
for nombre_archivo in os.listdir(RUTA_IMAGENES):
    ruta = os.path.join(RUTA_IMAGENES, nombre_archivo)
    if os.path.isfile(ruta) and nombre_archivo.lower().endswith(('.png', '.jpg', '.jpeg')):
        try:
            vector = extraer_vector(ruta)
            vectores.append(vector)
            nombres.append(nombre_archivo)
            print(f"> Procesada: {nombre_archivo}")
        except Exception as e:
            print(f"[ERROR] {nombre_archivo}: {e}")

# Guardamos como listas simples
data = {
    "vectores": vectores,
    "nombres": nombres
}

with open("vectores_imagenes.pkl", "wb") as f:
    pickle.dump(data, f)

print("✅ Archivo vectores_imagenes.pkl generado correctamente.")
