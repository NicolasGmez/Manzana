import os
import streamlit as st
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import tensorflow as tf

# Cargar el modelo (asegúrate de tener un archivo modelo.h5 en la carpeta)
modelo = tf.keras.models.load_model("modelooo.h5")

# Lista de nombres de clases (ajústala según tu modelo)
clases = ['Apple___scab', 'Apple___black_rot', 'Apple___rust', 'Apple___healthy',
    'Apple___alternaria_leaf_spot', 'Apple___brown_spot', 'Apple___gray_spot']

# Función para hacer la predicción
def predict_image(image):
    image = image.resize((256, 256))  # Ajustar tamaño
    image = np.array(image) / 255.0  # Normalizar
    image = np.expand_dims(image, axis=0)  # Añadir batch
    prediction = modelo.predict(image)
    
    class_index = np.argmax(prediction)  # Índice de la clase con mayor probabilidad
    class_name = clases[class_index]  # Nombre de la clase

    return class_name, prediction[0][class_index]  # Retorna el nombre y la probabilidad

# Interfaz gráfica en Streamlit
st.title("Identificador de enfermedades de hojas de manzana")
st.image("https://bosquenagal.com/wp-content/uploads/arbol-de-manzano2.jpg", use_column_width=True)

# Subir imagen o URL
upload_option = st.radio("Selecciona cómo subir la imagen:", ("Subir desde la PC", "Ingresar URL"))

image = None
if upload_option == "Subir desde la PC":
    uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
elif upload_option == "Ingresar URL":
    image_url = st.text_input("Ingresa la URL de la imagen")
    if image_url:
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))

if image:
    st.image(image, caption="Imagen cargada", use_column_width=True)
    
    # Botones de predicción
    col1, col2 = st.columns(2)
    if col1.button("Realizar predicción"):
        class_name, confidence = predict_image(image)
        st.write(f"### Predicción: **{class_name}**")
        st.write(f"📊 **Confianza:** {confidence:.2%}")  # Muestra la confianza en porcentaje
