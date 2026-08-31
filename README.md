# Clasificador de Reciclaje Inteligente

Proyecto Final de la materia Inteligencia Artificial.
Sistema basado en Transfer Learning que clasifica un objeto, reciclable, no reciclable y organico frente a la cámara web, mostrando el porcentaje de confianza en tiempo real.

## Problema que resuelve
La mala clasificación de residuos domésticos contamina los lotes de reciclaje. Este sistema ayuda a cualquier persona a saber, en segundos, a qué contenedor va un desecho.

## Modelo
- MobileNetV2 pre-entrenada en ImageNet (capas congeladas, Feature Extraction).
- Encabezado propio: `GlobalAveragePooling2D` + `Dropout(0.2)` + `Dense(3, softmax)`.
- Aumento de datos geométrico (volteo, rotación, zoom).
- Optimizador Adam (lr=0.001), pérdida `sparse_categorical_crossentropy`, 15 épocas.
- Accuracy de validación: 77.8% – 100% (mínimo exigido: 75%). Matriz de confusión final: 9/9 correctas.

## Dataset
45 imágenes (15 por clase): 26 fotografías propias tomadas en mi hogar + 19 imágenes de libre uso de internet. 
División: 36 entrenamiento / 9 validación (80/20, seed 42).

## Estructura de carpetas en tu Google Drive:

MyDrive/proyecto_reciclaje/
├── organico/        (15 fotos)
├── reciclable/      (15 fotos)
└── no_reciclable/   (15 fotos)


## Cómo ejecutarlo
1. Abrir el notebook `Proyecto_Reciclaje.ipynb` en Google Colab
2. Subir a Drive las 3 carpetas del dataset en `MyDrive/proyecto_reciclaje/`.
3. Presionar "Ejecutar todas" (monta Drive, carga los datos, entrena y evalúa el modelo).
4. Ejecutar la celda de Gradio: aparecerá un enlace público `*.gradio.live`. Al abrirlo, permitir la cámara, apuntar al objeto, capturar y presionar Submit.


## Resultados y análisis de errores
Con buena iluminación el sistema acierta (botella → reciclable 88%). Con poca luz puede dudar (rastrillo → empate 48/48), lo cual se analizó en el informe: el aumento de brillo agresivo se probó y descartó (33%), confirmando que con datasets pequeños el aumento debe ser moderado.

## Requisitos
Google Colab (TensorFlow y Gradio vienen incluidos). Para ejecución local: `pip install tensorflow gradio matplotlib`.

## Autoría
Proyecto individual de Diana Cabrera
Materia Inteligencia Artificial · 2026
