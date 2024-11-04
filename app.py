import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import unicodedata
from keras_facenet import FaceNet
from pymongo import MongoClient
from datetime import datetime
from PIL import Image
import cv2  # Importar OpenCV para tomar la foto

# Configuración de MongoDB
print("Configurando MongoDB...")
try:
    client = MongoClient("mongodb+srv://cesarcorrea:k9DexhefNDS9GTLs@cluster0.rwqzs.mongodb.net/AttendanceDB?retryWrites=true&w=majority&appName=Cluster0")
    db = client.AttendanceDB
    students_collection = db.students
    attendance_collection = db.attendance
    print("MongoDB configurado exitosamente.")
except Exception as e:
    print(f"Error conectando a MongoDB: {e}")

# Configuración de la API
API_URL = "https://face-recog-ml-ztsfu.eastus.inference.ml.azure.com/score"

# Configurar el modelo de FaceNet
print("Cargando modelo de FaceNet...")
try:
    embedder = FaceNet()
    print("Modelo de FaceNet cargado.")
except Exception as e:
    print(f"Error cargando FaceNet: {e}")

# Función para normalizar el nombre
def normalize_string(s):
    return unicodedata.normalize('NFD', s).encode('ascii', 'ignore').decode("utf-8").lower()

# Función para cargar datos de estudiantes desde MongoDB
def load_students_data():
    students_data = list(students_collection.find({}, {"_id": 0, "name": 1, "matricula": 1, "attendance": 1}))
    if students_data:
        return pd.DataFrame(students_data)
    else:
        return pd.DataFrame(columns=["name", "matricula", "attendance"])

# Función para limpiar la asistencia
def clear_attendance():
    students_collection.update_many({}, {"$set": {"attendance": False}})
    today = datetime.now()
    attendance_collection.delete_many({
        "date": {"$gte": today.replace(hour=0, minute=0, second=0, microsecond=0),
                 "$lt": today.replace(hour=23, minute=59, second=59, microsecond=999999)}
    })
    st.success("Asistencia borrada exitosamente.")

# Función para agregar un estudiante
def add_student(name, matricula):
    if students_collection.find_one({"matricula": matricula}):
        st.warning(f"El estudiante con matrícula {matricula} ya existe.")
    else:
        students_collection.insert_one({"name": name, "matricula": matricula, "attendance": False})
        st.success(f"Estudiante {name} agregado exitosamente.")

# Función para predecir usando la API de reconocimiento facial
def predict_image(image):
    img_array = np.array(image.convert("RGB"))
    detections = embedder.extract(img_array, threshold=0.95)
    if not detections:
        st.error("No se detectó ninguna cara en la imagen. Inténtalo de nuevo.")
        return None

    embedding = detections[0]["embedding"]

    # Preparar los datos en formato JSON para enviarlos a la API
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

# Función para tomar foto desde la cámara
def take_photo():
    cap = cv2.VideoCapture(0)
    st.info("Presiona 's' para tomar la foto y 'q' para salir.")
    img_path = None
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Presiona 's' para tomar la foto", frame)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            img_path = "captured_image.jpg"
            cv2.imwrite(img_path, frame)
            st.image(img_path, caption="Foto tomada", use_column_width=True)
            cap.release()
            cv2.destroyAllWindows()
            break
        elif cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break
    return img_path

# Interfaz de Streamlit
st.sidebar.title("Navegación")
page = st.sidebar.selectbox("Selecciona una página", ["Inicio", "Asistencia", "Reporte de Asistencia"])

# Página de inicio
if page == "Inicio":
    st.title("Sistema de Asistencia UPY")
    st.write("Bienvenido al Sistema de Asistencia. Usa la barra lateral para interactuar con la aplicación.")

# Página de asistencia
elif page == "Asistencia":
    st.title("Sistema de Asistencia")

    st.sidebar.subheader("Información de la Clase")
    career = st.sidebar.selectbox("Selecciona Carrera", ["Data Engineer", "Cybersecurity", "Embedded Systems", "Robotics"])
    quarter = st.sidebar.selectbox("Selecciona Cuatrimestre", ["Immersion", "Third Quarter", "Sixth Quarter", "Ninth Quarter"])
    group = st.sidebar.selectbox("Selecciona Grupo", ["A", "B"] if career == "Data Engineer" and quarter == "Ninth Quarter" else [])

    if group == "B":
        col1, col2 = st.columns([2, 1])

        # Columna izquierda: mostrar la tabla de estudiantes
        with col1:
            st.subheader("Datos de estudiantes para el grupo B")
            df_students = load_students_data()
            student_table = st.empty()  # Elemento para actualizar la tabla
            student_table.dataframe(df_students.sort_values(by='matricula'))

        # Columna derecha: opciones de actualización y carga de imagen
        with col2:
            st.subheader("Opciones")

            # Formulario para agregar un nuevo estudiante
            st.write("Agregar un nuevo estudiante")
            name = st.text_input("Nombre del Estudiante")
            matricula = st.text_input("Matrícula del Estudiante")
            
            if st.button("Agregar Estudiante"):
                if name and matricula:
                    add_student(name, matricula)
                else:
                    st.warning("Por favor, ingresa el nombre y la matrícula.")

            # Botón para actualizar la tabla de estudiantes
            if st.button("Actualizar Tabla"):
                df_students = load_students_data()  # Recargar datos
                student_table.dataframe(df_students.sort_values(by='matricula'))
                st.success("Tabla actualizada.")

            # Botón para tomar foto y usarla para identificar estudiante
            if st.button("Tomar Foto"):
                img_path = take_photo()
                if img_path:
                    image = Image.open(img_path)
                    detected_matricula = predict_image(image)
                    if detected_matricula:
                        # Actualizar asistencia en la base de datos
                        students_collection.update_one(
                            {"matricula": detected_matricula}, 
                            {"$set": {"attendance": True}}
                        )
                        attendance_collection.insert_one({
                            "name": detected_matricula, 
                            "timestamp": datetime.now()  # Fecha y hora combinadas
                        })
                        st.success(f"Asistencia marcada para la matrícula: {detected_matricula}")

            # Opción de subir imagen manualmente
            uploaded_image = st.file_uploader("Sube una imagen para identificar", type=["jpg", "png"])
            if uploaded_image:
                image = Image.open(uploaded_image)
                detected_matricula = predict_image(image)
                if detected_matricula:
                    # Actualizar asistencia en la base de datos
                    students_collection.update_one(
                        {"matricula": detected_matricula}, 
                        {"$set": {"attendance": True}}
                    )
                    attendance_collection.insert_one({
                        "name": detected_matricula, 
                        "timestamp": datetime.now()  # Fecha y hora combinadas
                    })
                    st.success(f"Asistencia marcada para la matrícula: {detected_matricula}")

            if st.button("Limpiar Asistencia"):
                clear_attendance()
                st.success("Asistencia limpiada correctamente.")

# Página de reporte de asistencia
elif page == "Reporte de Asistencia":
    st.title("Reporte de Asistencia")

    # Cargar y mostrar el reporte de asistencia
    df_attendance_report = load_students_data()
    if not df_attendance_report.empty:
        st.dataframe(df_attendance_report)
    else:
        st.write("No hay registros de asistencia para hoy.")
