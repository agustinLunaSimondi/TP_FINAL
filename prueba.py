# Instalar dependencias necesarias

# Importar bibliotecas
import os
from flask import Flask, request, render_template, send_from_directory
from flask_ngrok import run_with_ngrok
import numpy as np
from PIL import Image
import cv2
from tensorflow.keras.models import load_model

# Cargar los modelos preentrenados
UNET_MODEL_PATH = ''  # Cambia esto a tu modelo U-Net
CLASSIFICATION_MODEL_PATH = ''  # Cambia esto a tu modelo de clasificación

unet_model = load_model(UNET_MODEL_PATH)
classification_model = load_model(CLASSIFICATION_MODEL_PATH)

# Crear directorios para guardar imágenes
UPLOAD_FOLDER = './uploads'
MASK_FOLDER = './masks'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MASK_FOLDER, exist_ok=True)

# Crear la aplicación Flask
app = Flask(__name__)
run_with_ngrok(app)  # Activar Flask con ngrok

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MASK_FOLDER'] = MASK_FOLDER

def preprocesar_imagen(filepath, target_size=(256, 256)):
    """
    Preprocesar la imagen cargada para ser compatible con el modelo.
    """
    img = Image.open(filepath).resize(target_size)
    img_array = np.array(img) / 255.0  # Normalización
    return img, img_array

def generar_mascara(imagen, modelo):
    """
    Generar una máscara usando el modelo U-Net.
    """
    entrada = np.expand_dims(imagen, axis=0)  # Añadir batch dimension
    mascara = modelo.predict(entrada)[0]
    return mascara

def superponer_mascara(imagen, mascara, alpha=0.5):
    """
    Superponer la máscara sobre la imagen original.
    """
    mascara_color = np.zeros_like(imagen)
    mascara_color[..., 0] = (mascara * 255).astype(np.uint8)  # Canal rojo
    imagen_superpuesta = cv2.addWeighted(imagen, 1, mascara_color, alpha, 0)
    return imagen_superpuesta

def clasificar_con_mascara(imagen, mascara, modelo):
    """
    Clasificar la larva usando la imagen y la máscara generada.
    """
    combinada = np.concatenate([imagen, np.expand_dims(mascara, axis=-1)], axis=-1)
    combinada = np.expand_dims(combinada, axis=0)  # Añadir batch dimension
    prediccion = modelo.predict(combinada)[0]
    confianza = prediccion[0]
    return confianza

@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Página principal para cargar la imagen y mostrar resultados.
    """
    if request.method == 'POST':
        # Procesar archivo subido
        if 'file' not in request.files:
            return "No se seleccionó ningún archivo", 400
        file = request.files['file']
        if file.filename == '':
            return "No se seleccionó ningún archivo", 400
        
        # Guardar archivo subido
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        # Preprocesar la imagen
        img, img_array = preprocesar_imagen(filepath)
        
        # Generar máscara
        mascara = generar_mascara(img_array, unet_model)
        
        # Superponer máscara
        img_original = (img_array * 255).astype(np.uint8)
        img_masked = superponer_mascara(img_original, mascara)
        
        # Guardar la imagen con máscara superpuesta
        output_path = os.path.join(app.config['MASK_FOLDER'], f'masked_{file.filename}')
        Image.fromarray(img_masked).save(output_path)
        
        # Clasificar
        confianza = clasificar_con_mascara(img_array, mascara, classification_model)
        clase = "Joven" if confianza > 0.5 else "Vieja"
        confianza_porcentaje = confianza * 100 if confianza > 0.5 else (1 - confianza) * 100
        
        # Renderizar página con resultados
        return render_template('result.html',
                               original_image=file.filename,
                               masked_image=f'masked_{file.filename}',
                               clase=clase,
                               confianza=f"{confianza_porcentaje:.2f}%")
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

# Código para iniciar la aplicación Flask
if __name__ == '__main__':
    # Crear plantillas HTML
    with open('templates/index.html', 'w') as f:
        f.write("""
        <!doctype html>
        <title>Clasificación de Larvas</title>
        <h1>Sube una imagen para clasificar</h1>
        <form method="post" enctype="multipart/form-data">
            <input type="file" name="file">
            <input type="submit" value="Subir">
        </form>
        """)

    with open('templates/result.html', 'w') as f:
        f.write("""
        <!doctype html>
        <title>Resultados</title>
        <h1>Resultados</h1>
        <p><strong>Clasificación:</strong> {{ clase }}</p>
        <p><strong>Confianza:</strong> {{ confianza }}</p>
        <h2>Imagen Original</h2>
        <img src="{{ url_for('uploaded_file', filename=original_image) }}" style="max-width:500px;">
        <h2>Imagen con Máscara</h2>
        <img src="{{ url_for('masked_file', filename=masked_image) }}" style="max-width:500px;">
        <br><a href="/">Subir otra imagen</a>
        """)

    # Ejecutar la aplicación
    app.run()