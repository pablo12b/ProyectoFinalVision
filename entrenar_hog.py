import os
import glob
import cv2

# Rutas
POS_DIR = "dataset_hog/pos"
NEG_DIR = "dataset_hog/neg"

# 1. Generar lista de Positivas (pos.txt)
print("Generando pos.txt...")
with open("pos.txt", "w") as f:
    for path in glob.glob(os.path.join(POS_DIR, "*.jpg")):
        img = cv2.imread(path)
        if img is None: continue
        h, w = img.shape[:2]
        # Formato: ruta cantidad x y w h
        # Como ya son recortes, la persona ocupa todo (0,0,w,h)
        line = f"{path} 1 0 0 {w} {h}\n"
        f.write(line)

# 2. Generar lista de Negativas (neg.txt)
print("Generando neg.txt...")
with open("neg.txt", "w") as f:
    for path in glob.glob(os.path.join(NEG_DIR, "*.jpg")):
        f.write(f"{path}\n")

print("¡Listas generadas!")