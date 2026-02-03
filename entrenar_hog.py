import cv2
import numpy as np
import os
import glob
from sklearn.linear_model import SGDClassifier
from sklearn.utils import shuffle

# --- CONFIGURACIÓN ---
POS_DIR = "dataset_hog/pos"
NEG_DIR = "dataset_hog/neg"
BATCH_SIZE = 1000  # Procesar de a 1000 fotos para no llenar la RAM

print("--- CONFIGURANDO HOG ---")
winSize = (64, 128)
blockSize = (16, 16)
blockStride = (8, 8)
cellSize = (8, 8)
nbins = 9
hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)

# Usamos SGDClassifier con loss='hinge' (equivale a SVM Lineal)
# Esto permite usar .partial_fit() para entrenar por pedazos
model = SGDClassifier(loss='hinge', max_iter=1000, tol=1e-3, warm_start=True)

print("--- LEYENDO LISTA DE ARCHIVOS ---")
pos_files = glob.glob(os.path.join(POS_DIR, "*.jpg"))
neg_files = glob.glob(os.path.join(NEG_DIR, "*.jpg"))

# Etiquetas: 1 para persona, 0 para fondo
# Creamos pares (ruta, etiqueta)
dataset = [(f, 1) for f in pos_files] + [(f, 0) for f in neg_files]
dataset = shuffle(dataset, random_state=42) # Mezclar todo

total_samples = len(dataset)
print(f"Total de imágenes a procesar: {total_samples}")

# --- ENTRENAMIENTO POR LOTES (BATCHES) ---
for i in range(0, total_samples, BATCH_SIZE):
    batch_data = dataset[i : i + BATCH_SIZE]
    X_batch = []
    y_batch = []
    
    print(f"Procesando lote {i}/{total_samples}...")
    
    for path, label in batch_data:
        img = cv2.imread(path)
        if img is None: continue
        
        # SALVAVIDAS: Si la imagen no es 64x128, la redimensionamos a la fuerza
        if img.shape[1] != 64 or img.shape[0] != 128:
            img = cv2.resize(img, (64, 128))
            
        # Calcular HOG
        try:
            hist = hog.compute(img)
            if hist is None: continue
            X_batch.append(hist.flatten())
            y_batch.append(label)
        except:
            continue

    if len(X_batch) > 0:
        X_batch = np.array(X_batch)
        y_batch = np.array(y_batch)
        
        # Aquí ocurre la magia: partial_fit entrena sin olvidar lo anterior
        model.partial_fit(X_batch, y_batch, classes=[0, 1])

print("--- ENTRENAMIENTO FINALIZADO ---")

# --- EXPORTAR PARA C++ ---
print("Exportando modelo...")
coef = model.coef_.flatten()
intercept = model.intercept_

# Unir pesos y bias
svm_detector = np.hstack((coef, intercept))

# Guardar
fs = cv2.FileStorage("mi_hog_personalizado.yml", cv2.FILE_STORAGE_WRITE)
fs.write("svm_detector", svm_detector)
fs.release()

print("¡ÉXITO! Se generó 'mi_hog_personalizado.yml' sin explotar la RAM.")