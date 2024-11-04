import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import base64
from PIL import Image
from io import BytesIO
from keras_facenet import FaceNet
from pymongo import MongoClient
from datetime import datetime
import unicodedata

# Configuración de MongoDB
client = MongoClient("mongodb+srv://cesarcorrea:k9DexhefNDS9GTLs@cluster0.rwqzs.mongodb.net/AttendanceDB?retryWrites=true&w=majority&appName=Cluster0")
db = client.AttendanceDB
students_collection = db.students
attendance_collection = db.attendance

API_URL = "https://face-recog-ml-ztsfu.eastus.inference.ml.azure.com/score"  # URL de la API de reconocimiento facial

# Configurar el modelo de FaceNet
embedder = FaceNet()

# Función para predecir usando la API de reconocimiento facial
def predict_image(image):
    img_array = np.array(image.convert("RGB"))
    detections = embedder.extract(img_array, threshold=0.95)
    if not detections:
        st.error("No se detectó ninguna cara en la imagen. Inténtalo de nuevo.")
        return None

    embedding = detections[0]["embedding"]

    data = {"data": [embedding.tolist()]}
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {st.secrets["API_KEY"]}'
    }

    try:
        response = requests.post(API_URL, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        
        if isinstance(result, list) and len(result) > 0:
            matricula = result[0]  # Extrae la matrícula
            return matricula
        else:
            st.error("La API no devolvió ningún resultado válido.")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Error conectando con la API: {e}")
        return None

# Función para manejar la imagen Base64
def handle_uploaded_image(data_url):
    image_data = base64.b64decode(data_url.split(",")[1])
    image = Image.open(BytesIO(image_data))
    return image

# Interfaz para capturar foto desde la cámara usando HTML y JavaScript
st.markdown("""
    <h3>Tomar Foto desde la Cámara</h3>
    <p>Permite acceso a la cámara y toma una foto que se enviará a Streamlit.</p>
    <div>
        <button id="start-camera">Abrir Cámara</button>
        <video id="video" width="100%" autoplay style="display:none;"></video>
        <button id="click-photo" style="display:none;">Tomar Foto</button>
        <canvas id="canvas" style="display:none;"></canvas>
    </div>
    <script>
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const startCamera = document.getElementById('start-camera');
    const clickPhoto = document.getElementById('click-photo');
    
    startCamera.addEventListener('click', async function() {
        video.style.display = 'block';
        clickPhoto.style.display = 'block';
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
    });

    clickPhoto.addEventListener('click', function() {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
        const dataURL = canvas.toDataURL('image/jpeg');
        
        // Enviar la imagen a Streamlit usando JSON
        fetch('/send_image', {
            method: 'POST',
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({image_data: dataURL})
        }).then(response => {
            if (response.ok) {
                console.log('Imagen enviada exitosamente');
            } else {
                console.error('Error enviando la imagen');
            }
        });
    });
    </script>
""", unsafe_allow_html=True)

# Captura y procesamiento de la imagen
if 'image_data' in st.session_state:
    image = handle_uploaded_image(st.session_state['image_data'])
    st.image(image, caption="Foto tomada", use_column_width=True)
    
    # Realizar la predicción
    detected_matricula = predict_image(image)
    if detected_matricula:
        # Marcar asistencia en la base de datos
        students_collection.update_one(
            {"matricula": detected_matricula}, 
            {"$set": {"attendance": True}}
        )
        attendance_collection.insert_one({
            "name": detected_matricula, 
            "timestamp": datetime.now()
        })
        st.success(f"Asistencia marcada para la matrícula: {detected_matricula}")

# Barra lateral para seleccionar la clase
st.sidebar.title("Navegación")
career = st.sidebar.selectbox("Selecciona Carrera", ["Data Engineer", "Cybersecurity", "Embedded Systems", "Robotics"])
quarter = st.sidebar.selectbox("Selecciona Cuatrimestre", ["Immersion", "Third Quarter", "Sixth Quarter", "Ninth Quarter"])
group = st.sidebar.selectbox("Selecciona Grupo", ["A", "B"] if career == "Data Engineer" and quarter == "Ninth Quarter" else [])

if group == "B":
    col1, col2 = st.columns([2, 1])

    # Columna izquierda: mostrar la tabla de estudiantes
    with col1:
        st.subheader("Datos de estudiantes para el grupo B")
        df_students = pd.DataFrame(list(students_collection.find({}, {"_id": 0, "name": 1, "matricula": 1, "attendance": 1})))
        student_table = st.empty()
        student_table.dataframe(df_students.sort_values(by='matricula'))

    # Columna derecha: opciones de carga de imagen y agregar estudiante
    with col2:
        st.subheader("Opciones")

        # Formulario para agregar un nuevo estudiante
        st.write("Agregar un nuevo estudiante")
        name = st.text_input("Nombre del Estudiante")
        matricula = st.text_input("Matrícula del Estudiante")
        
        if st.button("Agregar Estudiante"):
            if name and matricula:
                students_collection.insert_one({"name": name, "matricula": matricula, "attendance": False})
                st.success(f"Estudiante {name} agregado exitosamente.")
            else:
                st.warning("Por favor, ingresa el nombre y la matrícula.")

        # Botón para actualizar la tabla de estudiantes
        if st.button("Actualizar Tabla"):
            df_students = pd.DataFrame(list(students_collection.find({}, {"_id": 0, "name": 1, "matricula": 1, "attendance": 1})))
            student_table.dataframe(df_students.sort_values(by='matricula'))
            st.success("Tabla actualizada.")

        # Opción de subir imagen manualmente para identificar
        uploaded_image = st.file_uploader("Sube una imagen para identificar", type=["jpg", "png"])
        if uploaded_image:
            image = Image.open(uploaded_image)
            detected_matricula = predict_image(image)
            if detected_matricula:
                students_collection.update_one(
                    {"matricula": detected_matricula}, 
                    {"$set": {"attendance": True}}
                )
                attendance_collection.insert_one({
                    "name": detected_matricula, 
                    "timestamp": datetime.now()
                })
                st.success(f"Asistencia marcada para la matrícula: {detected_matricula}")

        if st.button("Limpiar Asistencia"):
            students_collection.update_many({}, {"$set": {"attendance": False}})
            attendance_collection.delete_many({})
            st.success("Asistencia limpiada correctamente.")
