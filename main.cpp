#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <cstdlib> 

using namespace cv;
using namespace std;

// --- CONFIGURACIÓN ---
const int FRAME_WIDTH = 640;
const int FRAME_HEIGHT = 480;
// Reemplaza esto con tu URL real o localhost
const string SERVER_URL = "http://127.0.0.1:5000/detectar"; 

void enviarImagenAlBot(const Mat& img) {
    imwrite("temp_envio.jpg", img);
    // El "&" al final es vital: lanza el proceso en segundo plano para no congelar la cámara
    string command = "curl -s -X POST -F \"file=@temp_envio.jpg\" " + SERVER_URL + " &";
    system(command.c_str()); 
}

int main() {
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Error: Cámara no encontrada" << endl;
        return -1;
    }
    
    // Configuración de Hardware
    cap.set(CAP_PROP_FRAME_WIDTH, FRAME_WIDTH);
    cap.set(CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT);

    // --- 1. DETECTOR (HOG + SVM) ---
    HOGDescriptor hog; 
    
    // ESTRATEGIA: Usamos el detector pre-entrenado (INRIA)
    // Esto elimina el "cuadro verde fantasma" causado por el mal entrenamiento casero.
    // Además cumple con la ilustración del PDF que menciona HOG.
    hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
    
    cout << "--- SISTEMA HOG OPTIMIZADO (Speed + Accuracy) ---" << endl;

    Mat frame;
    int contadorDeteccion = 0;
    int cooldown = 0; 
    int frameCount = 0; 

    while (true) {
        cap >> frame;
        if (frame.empty()) break;
        frameCount++;

        // --- 2. OPTIMIZACIÓN DE FLUJO (Frame Skipping) ---
        // Procesamos HOG solo 1 de cada 3 frames.
        // Los otros 2 frames solo son de "paso" (video fluido).
        if (frameCount % 3 != 0) {
            imshow("Vigilancia", frame);
            if (waitKey(1) == 'q') break;
            continue; 
        }

        vector<Rect> found;
        vector<double> weights;

        // --- CALIBRACIÓN EQUILIBRADA ---
        // hitThreshold: 0.3 (Antes 1.1). Bajamos para que sea más sensible y te detecte fácil.
        // winStride: Size(8,8) (Antes 16,16). Volvemos al estándar para que no se salte píxeles.
        // groupThreshold: 2 (Antes 3). Exigimos menos confirmaciones.
        hog.detectMultiScale(frame, found, weights, 0.3, Size(8,8), Size(32,32), 1.05, 2);

        bool cuerpoEnteroDetectado = false;

        for (size_t i = 0; i < found.size(); i++) {
            Rect r = found[i];

            // --- FILTROS RELAJADOS ---
            
            // A. Altura: Aceptamos gente más pequeña (lejos) y más grande (cerca)
            if (r.height < 60 || r.height > 470) continue; 

            // B. Proporción: Aceptamos gente más "ancha" (hasta 0.85) por si usas ropa holgada
            double ratio = (double)r.width / r.height;
            if (ratio > 0.85 || ratio < 0.2) continue;

            // C. Bordes: Mantenemos este para evitar falsos al entrar
            if (r.x < 2 || r.y < 2 || (r.x + r.width) > frame.cols - 2) continue;

            // SI PASA: DIBUJAMOS
            rectangle(frame, r.tl(), r.br(), Scalar(0, 255, 0), 2);
            cuerpoEnteroDetectado = true;
            
            // DEBUG VISUAL: Esto te dirá qué está viendo el HOG
            // Si ves el número pero no el cuadro verde, es que el filtro lo borró.
            string info = to_string(r.height) + "px";
            putText(frame, info, Point(r.x, r.y - 5), FONT_HERSHEY_PLAIN, 1, Scalar(0,255,0), 1);
        }

        // --- LÓGICA DE CASCADA (HOG -> YOLO) ---
        if (cuerpoEnteroDetectado) {
            contadorDeteccion++;
        } else {
            contadorDeteccion = 0; 
        }

        // Se requieren 3 detecciones positivas seguidas (en frames procesados)
        // Como saltamos frames, esto equivale a unos 10 frames reales (~0.5 segs)
        if (contadorDeteccion >= 3 && cooldown == 0) {
            putText(frame, "ENVIANDO A YOLO...", Point(20, 50), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 255), 2);
            enviarImagenAlBot(frame); 
            contadorDeteccion = 0;
            cooldown = 40; // Esperar ~3 segundos antes de volver a enviar
        }

        if (cooldown > 0) cooldown--;

        imshow("Vigilancia", frame);
        if (waitKey(1) == 'q') break;
    }

    return 0;
}