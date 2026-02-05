# Proyecto Final: Sistema de Vigilancia Inteligente con HOG y YOLO

Este proyecto implementa un sistema de vigilancia híbrido que combina la eficiencia de **C++ con HOG+SVM** para la detección en tiempo real y la precisión de **Python con YOLOv11** para el análisis de posturas y notificaciones.

## 📂 Archivos del Proyecto
*   **`generar_dataset.py`**: Script para crear y aumentar el dataset de entrenamiento para HOG.
*   **`main.cpp`**: Aplicación cliente en C++ que captura video, detecta personas y envía alertas.
*   **`servidor.py`**: Servidor Flask en Python que recibe alertas, valida con YOLO y notifica por Telegram.

---

## 1. Dataset y Generación de Datos
Para mejorar la detección de personas mediante HOG, se realizó un proceso exhaustivo de curación y aumento de datos.

### Origen de Datos
Se utilizó un dataset base de estimación de posturas (`pose-estimation.v3i.yolov8`). Este formato contiene imágenes completas y archivos de etiquetas que indican la ubicación de las personas.

### Preprocesamiento (`generar_dataset.py`)
El script de generación realiza los siguientes pasos para crear muestras positivas de alta calidad:
1.  **Lectura y Parseo**: Lee las imágenes y sus correspondientes etiquetas YOLO (.txt).
2.  **Extracción (Cropping)**: Utiliza las coordenadas de las cajas delimitadoras (bounding boxes) para recortar únicamente a las personas de la escena.
3.  **Normalización**: Cada recorte se redimensiona a **64x128 píxeles**, que es el tamaño estándar requerido por el descriptor HOG de OpenCV.
4.  **Validación**: Se descartan recortes demasiado pequeños o mal formados para evitar ruido en el entrenamiento.

### Data Augmentation (Aumento de Datos)
Para cumplir con los requisitos de robustez y simular condiciones reales (como el movimiento de una cámara o desenfoque), se implementó un pipeline de aumentación utilizando la librería **Albumentations**. Por cada persona detectada, se generan **15 variaciones** incluyendo:
*   **Horizontal Flip**: Efecto espejo (p=0.5).
*   **Motion Blur**: Simulación de desenfoque por movimiento (clave para video en tiempo real).
*   **Rotación**: Leves rotaciones (±10 grados) para simular posturas naturales.
*   **Ruido Gaussiano**: Para robustecer el detector ante cámaras con 'grano' o baja luz.
*   **Brillo y Contraste**: Variaciones de iluminación.

**Resultado:** Un dataset robusto almacenado en `dataset_hog/pos` listo para entrenar o validar detectores HOG.

---

## 2. Documentación Técnica de Implementación

### 2.1. Cliente C++ (`main.cpp`) - Detector Local
Este módulo actúa como un agente de borde (Edge Agent), optimizado para bajo consumo de recursos y alta velocidad de respuesta.

#### Arquitectura y Librerías
*   **Core**: Utiliza `opencv2/objdetect.hpp` para el descriptor HOG y `opencv2/imgproc.hpp` para operaciones matriciales.
*   **Comunicación**: Invoca llamadas al sistema (`system()`) para ejecutar `curl`, permitiendo el envío asíncrono de datos (con `&`) para no bloquear el hilo de captura de video.

#### Lógica de Detección (HOG + SVM)
Se instancia un `HOGDescriptor` configurado con el **detector de personas por defecto (INRIA)** (`HOGDescriptor::getDefaultPeopleDetector()`). Esto carga los coeficientes del hiperplano de soporte de un SVM lineal pre-entrenado.

**Parámetros Críticos de `detectMultiScale`:**
*   **`hitThreshold` (0.3)**: Define el margen de tolerancia para la clasificación del SVM. Un valor más bajo aumenta el recall (detecta más) a costo de precisión. Se ajustó a 0.3 para maximizar la sensibilidad.
*   **`winStride` (8,8)**: El paso de la ventana deslizante. Un paso de 8px (mitad de celda) ofrece un equilibrio óptimo entre cobertura y coste computacional.
*   **`scale` (1.05)**: Factor de escalado para la pirámide de imágenes. El algoritmo reduce la imagen un 5% en cada nivel para detectar personas a diferentes distancias.
*   **`groupThreshold` (2)**: Requiere que al menos 2 rectángulos detectados se superpongan para considerar una detección válida, eliminando ruido esporádico.

#### Heurística de Filtrado Geométrico
Para reducir falsos positivos que el HOG pueda dejar pasar, se implementan filtros post-detección basados en la geometría esperada de un humano:
1.  **Filtro de Altura**: `60px < h < 470px`. Descarta objetos demasiado pequeños (lejos/ruido) o que ocupan toda la pantalla.
2.  **Ratio de Aspecto (Aspect Ratio)**: Se calcula `ratio = width / height`. Se aceptan solo detecciones con `0.2 < ratio < 0.85`, descartando objetos muy anchos (coches, muebles) o extremadamente delgados.
3.  **Supresión de Bordes**: Se ignoran detecciones que tocan los bordes del frame (`x < 2`), ya que los descriptores HOG incompletos suelen generar falsos positivos.

#### Máquina de Estados (Detección Temporal)
*   **Temporal Consistency**: Se implementa un contador (`contadorDeteccion`). Se requiere que el flujo HOG detecte una persona en **3 frames consecutivos procesados** (aprox. 0.5s reales) antes de activar una alerta.
*   **Cooldown**: Tras una alerta, el sistema entra en un estado de "enfriamiento" por 40 ciclos de loop, evitando ataques de denegación de servicio (DoS) hacia el servidor.

---

### 2.2. Servidor Python (`servidor.py`) - Inferencia de Alto Nivel
Implementado como un microservicio RESTful modular utilizando **Flask**, encargado de la validación semántica y la respuesta.

#### Pipeline de Procesamiento de Imagen
1.  **Ingesta en Memoria**:
    El endpoint `/detectar` recibe el archivo mediante `request.files`.
    *   *Técnica*: No se guarda en disco inmediatamente. Se lee el stream de bytes (`file.read()`) y se convierte a un buffer numpy (`np.frombuffer`), decodificándolo finalmente con `cv2.imdecode`. Esto reduce la latencia de I/O drásticamente.

#### Motor de Inferencia (Ultralytics YOLOv11)
Se utiliza el modelo `yolo11n-pose.pt` (Nano), cuantizado para inferencia rápida en CPU.
*   **Tarea**: Pose Estimation (Keypoint Detection).
*   **Inferencia**: `model.predict(img, conf=0.5)`. Se establece un umbral de confianza del 50%.
*   **Salida**: El modelo retorna un objeto `Results` que contiene:
    *   `boxes`: Bounding boxes de las personas.
    *   `keypoints`: Coordenadas (x,y) de 17 articulaciones (hombros, codos, rodillas, etc.).

#### Generación de Evidencia y Notificación
1.  **Visualización**: Se utiliza el método `.plot()` nativo de Ultralytics para renderizar el esqueleto sobre la imagen original.
2.  **Persistencia**: Se guardan dos versiones de la evidencia con timestamps precisos:
    *   `_org.jpg`: La imagen cruda enviada por el HOG (para auditoría).
    *   `_proc.jpg`: La imagen con el esqueleto superpuesto (para validación visual).
3.  **Integración con Telegram**:
    Se utiliza la librería `telebot` (pyTelegramBotAPI) en modo síncrono. Se envía la imagen procesada con un *caption* formateado como alerta de seguridad. El envío está encapsulado en un bloque `try-except` para garantizar que un fallo de red en la API de Telegram no tumbe el servicio de inferencia.

---

## 🚀 Cómo Ejecutar

1.  **Iniciar el Servidor (Cerebro):**
    ```bash
    python3 servidor.py
    ```
    *Asegúrate de tener configurado tu TOKEN y CHAT_ID de Telegram en el script.*

2.  **Iniciar el Cliente (Ojos):**
    ```bash
    # Compilar (asegúrate de tener CMake y OpenCV instalados)
    cmake .
    make
    # Ejecutar
    ./ProyectoFinal
    ```
