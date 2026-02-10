#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <cstdlib> 

using namespace cv;
using namespace std;

// CONFIGURACIÓN
const int FRAME_WIDTH = 640;
const int FRAME_HEIGHT = 480;
const string URL_FOTO = "http://127.0.0.1:5000/subir_foto"; 
const string URL_VIDEO = "http://127.0.0.1:5000/grabar_video"; 

// 1. Envía la foto (Bloqueante: Espera a que Python mande las 2 imágenes)
void enviarFoto(const Mat& img) {
    imwrite("temp_envio.jpg", img);
    cout << "\n[C++] Enviando evidencia fotográfica..." << endl;
    
    // SIN '&' al final -> El programa espera aquí
    string command = "curl -s -X POST -F \"file=@temp_envio.jpg\" " + URL_FOTO;
    system(command.c_str()); 
    cout << "[C++] Fotos enviadas." << endl;
}

// 2. Activa el video (Relevo de cámara)
void activarVideo(VideoCapture& cap) {
    cout << "[C++] Iniciando protocolo de video..." << endl;
    
    cap.release(); // Soltar cámara
    
    cout << "[C++] Grabando video en Python..." << endl;
    string command = "curl -s -X POST " + URL_VIDEO;
    system(command.c_str()); // Esperar a que termine de grabar y enviar
    
    // Recuperar cámara
    cap.open(0);
    if (!cap.isOpened()) {
        cerr << "Error Fatal: Cámara perdida." << endl;
        exit(-1);
    }
    cap.set(CAP_PROP_FRAME_WIDTH, FRAME_WIDTH);
    cap.set(CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT);
    cout << "[C++] Vigilancia restaurada.\n" << endl;
}

int main() {
    VideoCapture cap(0);
    if (!cap.isOpened()) return -1;
    
    cap.set(CAP_PROP_FRAME_WIDTH, FRAME_WIDTH);
    cap.set(CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT);

    HOGDescriptor hog; 
    hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
    
    cout << "--- SISTEMA VIGILANCIA COMPLETO ---" << endl;

    Mat frame;
    int contadorDeteccion = 0;
    int cooldown = 0; 
    int frameCount = 0; 

    while (true) {
        cap >> frame;
        if (frame.empty()) break;
        frameCount++;

        if (frameCount % 3 != 0) {
            imshow("Vigilancia C++", frame);
            if (waitKey(1) == 'q') break;
            continue; 
        }

        vector<Rect> found;
        vector<double> weights;

        // Detección
        hog.detectMultiScale(frame, found, weights, 0.6, Size(8,8), Size(32,32), 1.05, 2);

        bool cuerpoEnteroDetectado = false;
        for (size_t i = 0; i < found.size(); i++) {
            Rect r = found[i];
            
            // Filtros relajados
            if (r.height < 60) continue; 
            double ratio = (double)r.width / r.height;
            if (ratio < 0.25 || ratio > 0.9) continue;
            if (r.x < 2 || r.y < 2 || (r.x + r.width) > frame.cols - 2) continue;

            rectangle(frame, r.tl(), r.br(), Scalar(0, 255, 0), 2);
            cuerpoEnteroDetectado = true;
        }

        if (cuerpoEnteroDetectado) contadorDeteccion++;
        else contadorDeteccion = 0;

        // ALERTA
        if (contadorDeteccion >= 4 && cooldown == 0) {
            putText(frame, "ALERTA: GENERANDO REPORTE...", Point(20, 50), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 255), 2);
            imshow("Vigilancia C++", frame);
            waitKey(1);

            // SECUENCIA EXACTA:
            // 1. Enviar Foto (Python manda HOG y luego YOLO)
            enviarFoto(frame);

            // 2. Grabar Video (Python graba y manda)
            activarVideo(cap);
            
            contadorDeteccion = 0;
            cooldown = 100; // Pausa larga para no saturar Telegram
        }

        if (cooldown > 0) cooldown--;

        imshow("Vigilancia C++", frame);
        if (waitKey(1) == 'q') break;
    }

    return 0;
}