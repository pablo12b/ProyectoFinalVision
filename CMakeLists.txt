cmake_minimum_required(VERSION 3.16)
project(proyecto CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Indicar la ruta de OpenCV personalizada
set(OpenCV_DIR "/home/pablo/aplicaciones/Librerias/opencv/opencvi/lib/cmake/opencv4")

# Buscar OpenCV
find_package(OpenCV REQUIRED)

# Add your source files
add_executable(${PROJECT_NAME}
    main.cpp
)

# Incluir headers de OpenCV
target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS})

# Enlazar librerías de OpenCV
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})