import cv2
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

# --- CONFIGURACIÓN ---
POS_DIR = "dataset_hog/pos"  # Tus carpetas del dataset
NEG_DIR = "dataset_hog/neg"
CANTIDAD_PRUEBA = 200  # Probaremos con 200 fotos de cada tipo para no tardar horas

print("--- CARGANDO CONFIGURACIÓN HOG (Igual a C++) ---")
# Configuración idéntica a tu C++ optimizado
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

y_true = [] # La realidad (1=Persona, 0=Fondo)
y_pred = [] # Lo que dijo el HOG

print(f"--- INICIANDO TEST CON {CANTIDAD_PRUEBA} IMÁGENES POR CLASE ---")

# 1. PROBAR POSITIVOS (Debería detectar persona)
print("Procesando Positivas (Esperamos detecciones)...")
pos_imgs = glob.glob(os.path.join(POS_DIR, "*.jpg"))[:CANTIDAD_PRUEBA]

for path in pos_imgs:
    img = cv2.imread(path)
    if img is None: continue
    
    # Reducir un poco si son muy grandes, igual que en la cámara
    img = cv2.resize(img, (640, 480)) 
    
    # DETECCIÓN HOG (Parámetros del C++)
    # hitThreshold=0.3, winStride=(8,8), scale=1.05
    boxes, weights = hog.detectMultiScale(img, hitThreshold=0.3, winStride=(8,8), padding=(8,8), scale=1.05)
    
    y_true.append(1) # Es una persona
    
    if len(boxes) > 0:
        y_pred.append(1) # HOG dijo "Persona" (Acierto)
    else:
        y_pred.append(0) # HOG dijo "Nada" (Falso Negativo)

# 2. PROBAR NEGATIVOS (No debería detectar nada)
print("Procesando Negativas (Esperamos silencio)...")
neg_imgs = glob.glob(os.path.join(NEG_DIR, "*.jpg"))[:CANTIDAD_PRUEBA]

for path in neg_imgs:
    img = cv2.imread(path)
    if img is None: continue
    img = cv2.resize(img, (640, 480))
    
    boxes, weights = hog.detectMultiScale(img, hitThreshold=0.3, winStride=(8,8), padding=(8,8), scale=1.05)
    
    y_true.append(0) # No es persona
    
    if len(boxes) > 0:
        y_pred.append(1) # HOG dijo "Persona" (Falso Positivo)
    else:
        y_pred.append(0) # HOG dijo "Nada" (Acierto)

# --- CÁLCULOS ---
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred) # Precisión
sensitivity = recall_score(y_true, y_pred)  # Sensitividad (Recall)
specificity = tn / (tn + fp)                # Especificidad

print("\n" + "="*40)
print("       RESULTADOS FINALES")
print("="*40)
print(f"Total Evaluado: {len(y_true)} imágenes")
print(f"Verdaderos Positivos (TP): {tp}")
print(f"Falsos Positivos (FP):     {fp}")
print(f"Verdaderos Negativos (TN): {tn}")
print(f"Falsos Negativos (FN):     {fn}")
print("-" * 30)
print(f"👉 EXACTITUD (Accuracy):   {accuracy*100:.2f}%")
print(f"👉 PRECISIÓN:              {precision*100:.2f}%")
print(f"👉 SENSITIVIDAD (Recall):  {sensitivity*100:.2f}%")
print(f"👉 ESPECIFICIDAD:          {specificity*100:.2f}%")
print("="*40)

# --- GRAFICAR MATRIZ DE CONFUSIÓN ---
plt.figure(figsize=(6, 5))
cm_matrix = np.array([[tn, fp], [fn, tp]])
sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Pred: Nada', 'Pred: Persona'],
            yticklabels=['Real: Nada', 'Real: Persona'])
plt.ylabel('Realidad')
plt.xlabel('Predicción HOG')
plt.title('Matriz de Confusión HOG')
plt.savefig('matriz_confusion.png')
print("✅ Gráfico guardado como 'matriz_confusion.png'")

# --- GRAFICAR BARRAS DE MÉTRICAS ---
plt.figure(figsize=(8, 5))
metrics = ['Precisión', 'Sensitividad', 'Especificidad']
values = [precision, sensitivity, specificity]
colors = ['#4CAF50', '#2196F3', '#FF9800']

plt.bar(metrics, values, color=colors)
plt.ylim(0, 1.1)
for i, v in enumerate(values):
    plt.text(i, v + 0.02, f"{v*100:.1f}%", ha='center', fontweight='bold')

plt.title('Métricas de Rendimiento del Sistema')
plt.ylabel('Porcentaje (0-1)')
plt.savefig('grafico_metricas.png')
print("✅ Gráfico guardado como 'grafico_metricas.png'")

plt.show()