import cv2
import os
import glob
import albumentations as A
import numpy as np

# --- CONFIGURACIÓN CORREGIDA ---
# Apuntamos a las carpetas PADRE (train, valid, test). 
# El script solito añadirá "/images" y "/labels" después.
BASE_PATH = "/home/pablo/Documentos/Vision/ProyectoFinal/pose-estimation.v3i.yolov8"

RUTAS_DATASET = [
    os.path.join(BASE_PATH, "train"),
    os.path.join(BASE_PATH, "valid"),
    os.path.join(BASE_PATH, "test")
]

OUTPUT_FOLDER = "dataset_hog/pos"
META_POSITIVAS = 4200

# Crear carpeta de salida
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --- PIPELINE DE ALBUMENTATIONS (Requisito del PDF) ---
# Esto genera las variaciones de postura y movimiento
transform = A.Compose([
    A.HorizontalFlip(p=0.5),              # Espejo (muy útil para posturas)
    A.RandomBrightnessContrast(p=0.2),    # Cambios de luz
    A.Rotate(limit=10, p=0.4),            # Rotación leve (simula caminar chueco/correr)
    A.MotionBlur(blur_limit=5, p=0.2),    # Simula movimiento (clave para el PDF)
    # Ruido Gaussiano genérico para evitar errores de versión
    A.GaussNoise(p=0.2),                  
])

print(f"--- Iniciando extracción de peatones ---")
print(f"Meta: {META_POSITIVAS} imágenes de 64x128")

count = 0

# Recorremos train, valid y test para sacar hasta la última persona
for split in RUTAS_DATASET:
    img_dir = os.path.join(split, "images")
    lbl_dir = os.path.join(split, "labels")
    
    # Verificar que existan las carpetas
    if not os.path.exists(img_dir) or not os.path.exists(lbl_dir):
        print(f"Saltando {split} (no encontrado)...")
        continue

    # Listar imágenes jpg y png
    files = glob.glob(os.path.join(img_dir, "*.jpg")) + glob.glob(os.path.join(img_dir, "*.png"))
    print(f"Procesando carpeta '{split}' con {len(files)} fotos...")

    for img_path in files:
        if count >= META_POSITIVAS: break
        
        # Buscar el .txt correspondiente
        filename = os.path.basename(img_path)
        name_only = os.path.splitext(filename)[0]
        txt_path = os.path.join(lbl_dir, name_only + ".txt")
        
        if not os.path.exists(txt_path): continue
        
        # Cargar imagen original
        img = cv2.imread(img_path)
        if img is None: continue
        h_img, w_img, _ = img.shape

        # Leer etiquetas YOLO (class x y w h)
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            if count >= META_POSITIVAS: break
            
            parts = line.strip().split()
            # Asumimos que todas las clases del dataset de Pose son personas (class 0)
            
            # Matemática para convertir YOLO (0-1) a Píxeles
            # center_x, center_y, width, height
            rel_x, rel_y, rel_w, rel_h = map(float, parts[1:5])
            
            # Calcular esquinas
            w_box = int(rel_w * w_img)
            h_box = int(rel_h * h_img)
            x_center = int(rel_x * w_img)
            y_center = int(rel_y * h_img)
            
            x1 = int(x_center - w_box / 2)
            y1 = int(y_center - h_box / 2)
            x2 = x1 + w_box
            y2 = y1 + h_box
            
            # Corregir coordenadas si se salen de la imagen
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w_img, x2)
            y2 = min(h_img, y2)
            
            # Validar recorte mínimo (para no guardar basura)
            if (x2 - x1) < 20 or (y2 - y1) < 40: continue
            
            # --- 1. EXTRAER RECORTE ---
            crop = img[y1:y2, x1:x2]
            
            try:
                # --- 2. REDIMENSIONAR A 64x128 (ESTÁNDAR HOG) ---
                crop_resized = cv2.resize(crop, (64, 128))
                
                # Guardar el recorte original limpio
                cv2.imwrite(f"{OUTPUT_FOLDER}/pos_{count}.jpg", crop_resized)
                count += 1
                
                # --- 3. GENERAR COPIAS AUMENTADAS (DATA AUGMENTATION) ---
                # Generamos 3 versiones extra por cada persona encontrada
                # Esto nos ayuda a llegar a 4000 rápido y cumple el requisito del PDF
                for _ in range(15):
                    if count >= META_POSITIVAS: break
                    
                    augmented = transform(image=crop_resized)["image"]
                    cv2.imwrite(f"{OUTPUT_FOLDER}/pos_aug_{count}.jpg", augmented)
                    count += 1
                    
            except Exception as e:
                continue

print(f"¡Terminado! Total de positivas generadas: {count}")