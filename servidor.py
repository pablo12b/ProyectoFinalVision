from flask import Flask, request
import cv2
import time
import os
from ultralytics import YOLO
import telebot

# --- CONFIGURACIÓN ---
TOKEN = "8298649145:AAHuZmXFg6nXKPX-6jiR5bmnLpWkaNNHJ2U"
CHAT_ID = "1903609826"
model = YOLO('yolo11n-pose.pt')

app = Flask(__name__)
bot = telebot.TeleBot(TOKEN)

# --- RUTA 1: GESTIÓN DE FOTOS (HOG + YOLO) ---
@app.route('/subir_foto', methods=['POST'])
def recibir_foto():
    try:
        print("📸 [FOTO] Recibida desde C++...")
        file = request.files['file']
        
        # Guardamos la imagen original (que ya trae el cuadro verde de C++)
        filename_hog = "evidencia_hog.jpg"
        file.save(filename_hog)
        
        # --- ENVÍO 1: FOTO HOG ORIGINAL ---
        print("   -> Enviando Foto HOG...")
        with open(filename_hog, 'rb') as foto:
            bot.send_photo(CHAT_ID, foto, caption="1️⃣ Detección Inicial (HOG - C++)")

        # --- ENVÍO 2: PROCESAMIENTO YOLO ---
        print("   -> Procesando YOLO...")
        img = cv2.imread(filename_hog)
        
        # Aplicar IA de Postura
        results = model(img)
        img_yolo = results[0].plot() # Dibuja el esqueleto
        
        # Guardar y Enviar
        filename_yolo = "evidencia_yolo.jpg"
        cv2.imwrite(filename_yolo, img_yolo)
        
        with open(filename_yolo, 'rb') as foto:
            bot.send_photo(CHAT_ID, foto, caption="2️⃣ Análisis de Postura (YOLO - Python)")
        
        print("✅ [FOTOS] Ambas imágenes enviadas.")
        return "Fotos OK", 200

    except Exception as e:
        print(f"❌ Error en Fotos: {e}")
        return "Error", 500

# --- RUTA 2: GRABACIÓN DE VIDEO ---
@app.route('/grabar_video', methods=['POST'])
def grabar_video():
    try:
        print("🎥 [VIDEO] Iniciando grabación de 5s...")
        
        # ABRIR CÁMARA (C++ ya la soltó)
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Error: Cámara ocupada o no encontrada.")
            return "Error Camara", 500

        # Configuración Video
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = 20.0
        output_file = "evidencia_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

        start_time = time.time()
        
        # Grabar 5 segundos
        while (time.time() - start_time) < 5:
            ret, frame = cap.read()
            if not ret: break
            
            # Dibujar esqueleto en el video también
            results = model(frame, verbose=False)
            annotated = results[0].plot()
            out.write(annotated)

        cap.release()
        out.release()
        
        # --- ENVÍO 3: VIDEO ---
        print("🚀 [VIDEO] Enviando a Telegram...")
        with open(output_file, 'rb') as video:
            bot.send_video(CHAT_ID, video, caption="3️⃣ Video de Comportamiento (5s)")
        
        print("✅ [VIDEO] Enviado.")
        return "Video OK", 200
        
    except Exception as e:
        print(f"❌ Error en Video: {e}")
        return "Error", 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=False)