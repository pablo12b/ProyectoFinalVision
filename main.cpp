#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <fstream> 
#include <unistd.h> 

using namespace cv;
using namespace std;

// --- CONFIGURACIÓN ---
const int FRAME_WIDTH = 640;
const int FRAME_HEIGHT = 480;
// Rutas del Servidor Python
const string URL_FOTO = "http://127.0.0.1:5000/subir_foto"; 
const string URL_VIDEO = "http://127.0.0.1:5000/grabar_video"; 

// --- CUMPLIMIENTO RÚBRICA: MEDIR RAM ---
double getRAMUsageMB() {
    long rss = 0L;
    FILE* fp = NULL;
    if ((fp = fopen("/proc/self/statm", "r")) == NULL) return 0.0;
    if (fscanf(fp, "%*s%ld", &rss) != 1) { fclose(fp); return 0.0; }
    fclose(fp);
    return (double)rss * (double)sysconf(_SC_PAGESIZE) / 1024.0 / 1024.0;
}

void enviarFoto(const Mat& img) {
    imwrite("temp_envio.jpg", img);
    // Enviar foto vía API REST (POST)
    string command = "curl -s -X POST -F \"file=@temp_envio.jpg\" " + URL_FOTO;
    system(command.c_str()); 
}

void activarVideo(VideoCapture& cap) {
    cap.release(); // Soltar cámara
    // Ordenar grabación vía API REST
    string command = "curl -s -X POST " + URL_VIDEO;
    system(command.c_str()); 
    // Recuperar cámara
    cap.open(0);
    cap.set(CAP_PROP_FRAME_WIDTH, FRAME_WIDTH);
    cap.set(CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT);
}

int main() {
    VideoCapture cap(0);
    if (!cap.isOpened()) return -1;
    
    cap.set(CAP_PROP_FRAME_WIDTH, FRAME_WIDTH);
    cap.set(CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT);

    // --- CONFIGURACIÓN HOG ---
    HOGDescriptor hog; 
    hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
    
    cout << "--- SISTEMA ACTIVO: MODO SENSIBLE + METRICAS ---" << endl;

    Mat frame;
    int contadorDeteccion = 0;
    int cooldown = 0; 
    int frameCount = 0; 
    
    // Variables para FPS
    double fps = 0;
    int fpsFrames = 0;
    double fpsTime = 0;

    while (true) {
        double startTick = (double)getTickCount(); // Inicio cronómetro

        cap >> frame;
        if (frame.empty()) break;
        frameCount++;

        if (frameCount % 3 != 0) {
            // Mostrar métricas siempre, aunque saltemos frames
            string info = "FPS: " + to_string((int)fps) + " | RAM: " + to_string((int)getRAMUsageMB()) + " MB";
            putText(frame, info, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 0, 0), 2);
            imshow("Vigilancia C++ (HOG)", frame);
            if (waitKey(1) == 'q') break;
            continue; 
        }

        vector<Rect> found;
        vector<double> weights;

        // --- CORRECCIÓN CLAVE: PARÁMETROS SENSIBLES ---
        // hitThreshold: 0.1 (Muy sensible, te detectará sentado)
        // winStride: (8,8) Precisión estándar
        // padding: (16,16) Ayuda si estás cerca de los bordes
        hog.detectMultiScale(frame, found, weights, 0.1, Size(8,8), Size(16,16), 1.05, 2);

        bool cuerpoEnteroDetectado = false;

        for (size_t i = 0; i < found.size(); i++) {
            Rect r = found[i];

            // --- FILTROS RELAJADOS (Para que te detecte sentado) ---
            
            // 1. Altura: Solo ignoramos ruido muy pequeño
            if (r.height < 50) continue; 

            // 2. Proporción: 
            // - Ignoramos palos muy finos (< 0.2)
            // - Aceptamos "cuadrados" anchos hasta 1.2 (porque sentado eres ancho)
            double ratio = (double)r.width / r.height;
            if (ratio < 0.2 || ratio > 1.2) continue;

            // 3. Bordes:
            if (r.x < 2 || r.y < 2 || (r.x + r.width) > frame.cols - 2) continue;

            // DIBUJAR DETECCIÓN
            rectangle(frame, r.tl(), r.br(), Scalar(0, 255, 0), 2);
            
            // RÚBRICA: Mostrar Confianza (Weights)
            // weights[i] es el nivel de confianza del algoritmo
            string label = "Conf: " + to_string(weights[i]).substr(0,4);
            putText(frame, label, Point(r.x, r.y - 5), FONT_HERSHEY_PLAIN, 1, Scalar(0,255,0), 1);
            
            cuerpoEnteroDetectado = true;
        }

        // --- RÚBRICA: INFORMACIÓN EN PANTALLA ---
        string info = "FPS: " + to_string((int)fps) + " | RAM: " + to_string((int)getRAMUsageMB()) + " MB";
        putText(frame, info, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 0, 0), 2);

        // LÓGICA DE ALERTA
        if (cuerpoEnteroDetectado) contadorDeteccion++;
        else contadorDeteccion = 0;

        if (contadorDeteccion >= 4 && cooldown == 0) {
            putText(frame, "ENVIANDO API...", Point(20, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
            imshow("Vigilancia C++ (HOG)", frame);
            waitKey(1); 

            enviarFoto(frame);
            activarVideo(cap);
            
            contadorDeteccion = 0;
            cooldown = 100; 
        }

        if (cooldown > 0) cooldown--;

        // CÁLCULO FPS
        double endTick = (double)getTickCount();
        fpsTime += (endTick - startTick) / getTickFrequency();
        fpsFrames++;

        if (fpsFrames >= 5) {
            fps = 5.0 / fpsTime;
            fpsFrames = 0;
            fpsTime = 0;
        }

        imshow("Vigilancia C++ (HOG)", frame);
        if (waitKey(1) == 'q') break;
    }

    return 0;
}