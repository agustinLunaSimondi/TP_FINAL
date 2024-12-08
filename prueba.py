import os
from flask import Flask, request, render_template, send_from_directory
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import tensorflow as tf
import uuid  # Para generar nombres únicos
import matplotlib.pyplot as plt
import base64

# Configuración de la aplicación
app = Flask(__name__)

# Rutas de los modelos
UNET_MODEL_PATH = 'modelos/unet_model.h5'
CLASSIFICATION_MODEL_PATH = 'modelos/model_dropout.keras'

# Directorios para guardar imágenes
UPLOAD_FOLDER = './uploads'
MASK_FOLDER = './masks'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MASK_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MASK_FOLDER'] = MASK_FOLDER

# Cargar modelos
unet_model = load_model(UNET_MODEL_PATH, compile=False)
classification_model = load_model(CLASSIFICATION_MODEL_PATH)

def apply_morphological_operations(mask, min_contour_area=50, target_size=(128, 128)):
     #Asegurarse de que la máscara sea binaria y de tipo uint8
    mask = (mask > 0.5).astype(np.uint8) * 255


    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=2)

    center = (mask.shape[1] // 2, mask.shape[0] // 2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_mask = np.zeros_like(mask)

    if contours:
        min_dist = float('inf')
        closest_contour = None

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_contour_area:
                continue

            M = cv2.moments(contour)
            if M['m00'] == 0:
                continue
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            dist = np.sqrt((cx - center[0]) ** 2 + (cy - center[1]) ** 2)

            if dist < min_dist:
                min_dist = dist
                closest_contour = contour

        if closest_contour is not None:
            cv2.drawContours(final_mask, [closest_contour], -1, 255, thickness=cv2.FILLED)

    final_mask = cv2.resize(final_mask, target_size, interpolation=cv2.INTER_NEAREST)
    kernel = np.ones((1, 1), np.uint8)
    final_mask = cv2.dilate(final_mask, kernel, iterations=7)

    kernel_elipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel_elipse, iterations=3)

    return final_mask


def preprocesar_imagen(filepath, target_size=(256, 256)):
    """
    Preprocesar la imagen cargada para ser compatible con el modelo.
    Asegura que la imagen esté en formato RGB y normaliza los valores de píxeles.
    """
    # Leer la imagen con OpenCV
    img = cv2.imread(filepath)
    if img is None:
        raise ValueError("No se pudo cargar la imagen.")
    
    # Convertir a RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Redimensionar la imagen
    img = cv2.resize(img, target_size)
    
    # Convertir a array de NumPy y normalizar
    img_array = img.astype(np.float32) / 255.0
    
    # Confirmar dimensiones de la matriz (debería ser [altura, ancho, 3])
    if img_array.shape[-1] != 3:
        raise ValueError(f"La imagen no tiene 3 canales. Formato actual: {img_array.shape}")
    
    return img, img_array

def generar_mascara(imagen, modelo):
    """
    Generar una máscara usando el modelo U-Net.
    """
    imagen = cv2.cvtColor(imagen, cv2.COLOR_RGB2BGR)
    imagen_resized = tf.image.resize(imagen, (128, 128))
    entrada = np.expand_dims(imagen_resized, axis=0)
    mascara = modelo.predict(entrada)[0]
    # Redimensionar la máscara a 256x256 para la superposición
    mascara = apply_morphological_operations(mascara)
    mascara_resized = cv2.resize(mascara, (256, 256))
    return (mascara_resized > 0.5).astype(np.uint8)  # Binarizar máscara

def superponer_mascara(imagen, mascara, alpha=0.5):
    """
    Superponer la máscara sobre la imagen original con transparencia y tonalidad roja.
    """
    # Asegurar que la máscara tiene el mismo tamaño que la imagen
    mascara_resized = cv2.resize(mascara, (imagen.shape[1], imagen.shape[0]))
    

    # Crear una máscara RGB con tonalidad roja
    mascara_rgb = np.zeros_like(imagen, dtype=np.uint8)
    mascara_rgb[:, :, 2] = (mascara_resized * 255).astype(np.uint8)  # Canal rojo (BGR, donde R es el índice 2)

    # Combinar la imagen y la máscara con transparencia
    imagen_superpuesta = cv2.addWeighted(imagen, 1 - alpha, mascara_rgb, alpha, 0)
    return imagen_superpuesta , mascara_rgb

def clasificar_imagen(imagen, mascara, modelo):
    """
    Clasificar la larva usando la imagen procesada.
    """
    # Asegurar que la máscara es binaria y tiene un canal
    if mascara.ndim != 2:
        mascara = np.squeeze(mascara, axis=-1)
    
    # Crear máscara en RGB
   
    mascara_rgb = cv2.merge([mascara * 255, mascara * 255, mascara * 255]).astype(np.uint8)

    #mascara_rgb = cv2.merge([mascara, mascara, mascara]).astype(np.uint8)
    
    # Aplicar la máscara a la imagen original
    imagen_resized = cv2.resize(imagen, (256, 256))
    entrada = cv2.bitwise_and(imagen_resized, mascara_rgb)
    # Expandir dimensión para compatibilidad con el modelo
    entrada_expandid = np.expand_dims(entrada, axis=0)

    # Obtener predicción
    prediccion = modelo.predict(entrada_expandid)[0]
    return prediccion , entrada

@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Página principal para cargar la imagen y mostrar resultados.
    """
    if request.method == 'POST':
        if 'file' not in request.files or request.files['file'].filename == '':
            return "No se seleccionó ningún archivo", 400
        file = request.files['file']
        
        # Guardar archivo subido con un nombre único
        filename = f"{uuid.uuid4().hex}_{file.filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Preprocesar la imagen
        img, img_array = preprocesar_imagen(filepath)
        
        # Generar máscara
        mascara = generar_mascara(img_array, unet_model)
        
        # Superponer máscara sobre la imagen original
        img_original = (img_array * 255).astype(np.uint8)
        img_masked , sup = superponer_mascara(img_original, mascara)
        
        # Guardar la imagen con máscara superpuesta
        masked_filename = f"masked_{filename}"
        masked_path = os.path.join(app.config['MASK_FOLDER'], masked_filename)
        cv2.imwrite(masked_path, img_masked)
        
        # Clasificar
        prediccion,enmascarada= clasificar_imagen(img_original, mascara, classification_model)
        clase = "Apta" if prediccion > 0.5 else "No Apta"
        confianza = prediccion * 100 if prediccion > 0.5 else (1 - prediccion) * 100

        
        
        # Codificar la imagen procesada para mostrarla en la página
        _, buffer = cv2.imencode('.png', enmascarada)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Renderizar página con resultados
        return render_template(
            'result.html',
            original_image=filename,
            masked_image=masked_filename,
            processed_image=img_base64,
            clase=clase,
            confianza=f"{confianza}"
        )
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """
    Servir archivos subidos.
    """
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/masks/<filename>')
def masked_file(filename):
    """
    Servir imágenes con máscara.
    """
    return send_from_directory(app.config['MASK_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
