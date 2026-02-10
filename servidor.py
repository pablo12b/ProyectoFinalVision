from flask import Flask, request
import cv2
import time
import os
import psutil  # Para la rúbrica (RAM)
from ultralytics import YOLO
import telebot

# --- CONFIGURACIÓN ---
TOKEN = "8298649145:AAHuZmXFg6nXKPX-6jiR5bmnLpWkaNNHJ2U"
model = YOLO('yolo11n-pose.pt')

app = Flask(__name__)
bot = telebot.TeleBot(TOKEN)

# Base de datos de usuarios en memoria
suscriptores = set()

# --- GESTIÓN DE USUARIOS (RÚBRICA) ---
@bot.message_handler(commands=['start'])
def suscribir(message):
    cid = message.chat.id
    suscriptores.add(cid)
    bot.reply_to(message, f"✅ Sistema Configurado.\nID: {cid}\nRecibirás: HOG + YOLO + VIDEO")
    print(f"➕ Usuario suscrito: {cid}")

# --- FUNCIÓN DE MÉTRICAS (RÚBRICA) ---
def obtener_datos_rubrica():
    ram = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    return f"RAM Server: {ram:.1f}MB"

# --- RUTA 1: FOTOS (DOBLE ENVÍO: HOG + YOLO) ---
@app.route('/subir_foto', methods=['POST'])
def recibir_foto():
    if not suscriptores:
        print("⚠️ Alerta recibida pero no hay usuarios (/start).")
        return "Sin usuarios", 200

    try:
        print("📸 Recibiendo evidencia...")
        file = request.files['file']
        filename_hog = "evidencia_hog.jpg"
        file.save(filename_hog) # Guardamos la original que viene de C++

        # --- 1. ENVIAR FOTO ORIGINAL (HOG) ---
        print("   -> Enviando Evidencia 1 (HOG)...")
        with open(filename_hog, 'rb') as foto:
            for uid in suscriptores:
                try:
                    bot.send_photo(uid, foto, caption="1️⃣ Detección Inicial (C++ HOG)")
                except: pass

        # --- 2. PROCESAR Y ENVIAR FOTO YOLO (CON MÉTRICAS) ---
        print("   -> Procesando Evidencia 2 (YOLO)...")
        img = cv2.imread(filename_hog)
        results = model(img)
        
        # Extraer métricas para la rúbrica
        confianza = results[0].boxes.conf.mean().item() if len(results[0].boxes) > 0 else 0.0
        puntos = results[0].keypoints.xy.shape[1] if results[0].keypoints is not None else 0
        info_ram = obtener_datos_rubrica()
        
        # Dibujar esqueleto
        img_yolo = results[0].plot()

        # ESTAMPAR DATOS EN LA IMAGEN (Requisito Rúbrica)
        texto = f"Conf: {confianza:.2f} | Ptos: {puntos} | {info_ram}"
        # Fondo negro para el texto para que se lea bien
        cv2.rectangle(img_yolo, (5, 5), (600, 40), (0,0,0), -1)
        cv2.putText(img_yolo, texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, (0, 255, 255), 2)
        
        filename_yolo = "evidencia_yolo.jpg"
        cv2.imwrite(filename_yolo, img_yolo)

        print("   -> Enviando Evidencia 2 (YOLO)...")
        with open(filename_yolo, 'rb') as foto:
            for uid in suscriptores:
                try:
                    bot.send_photo(uid, foto, caption=f"2️⃣ Análisis IA (Python)\n📊 {texto}")
                except: pass
        
        return "Fotos OK", 200

    except Exception as e:
        print(f"❌ Error Fotos: {e}")
        return "Error", 500

# --- RUTA 2: VIDEO (EVIDENCIA 3) ---
@app.route('/grabar_video', methods=['POST'])
def grabar_video():
    if not suscriptores: return "Sin usuarios", 200

    try:
        print("🎥 Grabando Video de 5s...")
        # C++ ya soltó la cámara, la tomamos nosotros
        cap = cv2.VideoCapture(0)
        if not cap.isOpened(): return "Error Camara", 500

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        output_file = "evidencia_video.mp4"
        # Codec H264 o mp4v
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_file, fourcc, 20.0, (width, height))

        start = time.time()
        while (time.time() - start) < 5:
            ret, frame = cap.read()
            if not ret: break
            
            # Dibujar esqueleto en el video también (Opcional, se ve genial)
            results = model(frame, verbose=False)
            annotated = results[0].plot()
            out.write(annotated)

        cap.release()
        out.release()

        # --- 3. ENVIAR VIDEO ---
        print("   -> Enviando Evidencia 3 (Video)...")
        with open(output_file, 'rb') as video:
            for uid in suscriptores:
                try:
                    bot.send_video(uid, video, caption="3️⃣ Registro de Movimiento (5s)")
                except: pass

        return "Video OK", 200

    except Exception as e:
        print(f"❌ Error Video: {e}")
        return "Error", 500

def iniciar_bot():
    bot.infinity_polling()

if __name__ == '__main__':
    # Hilo para que el bot responda a /start mientras Flask escucha a C++
    import threading
    t = threading.Thread(target=iniciar_bot)
    t.start()
    
    app.run(host='0.0.0.0', port=5000, threaded=False)