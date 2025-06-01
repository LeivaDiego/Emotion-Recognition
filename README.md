
# Proyecto 4: Detección de Expresiones Faciales en Tiempo Real

**Universidad del Valle de Guatemala**  
**Facultad de Ingeniería**  
**Departamento de Ciencias de la Computación**  
**Curso: Visión por Computadora**

## Integrantes

- Diego Leiva – 21752  
- María Ramírez – 21342  
- Gustavo González – 21438  
- Pablo Orellana – 21970  

## Descripción del Proyecto

Este proyecto consiste en un sistema de reconocimiento de emociones faciales en tiempo real utilizando MediaPipe para detección y extracción de características faciales, y un modelo de aprendizaje profundo (MLP) entrenado sobre el dataset FER+ para clasificar emociones básicas.

El flujo principal del sistema es:

1. Captura de video en tiempo real desde la webcam.
2. Detección de rostro usando MediaPipe.
3. Extracción de 3D face landmarks (478 puntos).
4. Clasificación de emociones con un modelo preentrenado en PyTorch.
5. Visualización de resultados con bounding box y etiqueta de la emoción.

## Tecnologías Utilizadas

- **Python** `3.12.9`
- **PyTorch** para entrenamiento e inferencia del modelo
- **MediaPipe** para detección facial y extracción de landmarks
- **OpenCV** para procesamiento y visualización de video
- **scikit-learn** para métricas de evaluación

## Dataset Utilizado

- **FER+** – Dataset de expresiones faciales etiquetado con múltiples anotadores humanos  
  🔗 [FER+ en Kaggle](https://www.kaggle.com/datasets/subhaditya/fer2013plus/data)

## Modelos Preentrenados de MediaPipe

- **Detector facial MediaPipe**:  
  🔗 [BlazeFace Short-Range](https://ai.google.dev/edge/mediapipe/solutions/vision/face_detector#blazeface_short-range)

- **Face Landmark Model**:  
  🔗 [FaceLandmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker#models)

## Cómo Ejecutar

1. Instala las dependencias:

    ```bash
    pip install -r requirements.txt
    ```

2. Ejecuta el script principal:

   ```bash
   python src/live_inference.py
   ```

> [!NOTE]
> Asegúrate de tener conectada tu cámara web y los modelos `.pth` y `.tflite` descargados en las rutas indicadas.

## Resultados y Evaluación

El modelo fue entrenado con FER+ y alcanza resultados aceptables en emociones como felicidad, sorpresa y enojo.
Durante la inferencia en tiempo real, se logra una visualización fluida (\~30 FPS en GPU) con predicciones en vivo y colores diferenciados por emoción.
