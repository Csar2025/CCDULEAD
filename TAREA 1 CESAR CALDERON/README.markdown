# Análisis de Sentimientos para Reseñas de Clientes

## Objetivo
Este proyecto implementa un modelo simple de análisis de sentimientos binario para clasificar reseñas de clientes como positivas o negativas. Utiliza scikit-learn para el entrenamiento del modelo y NLTK para el preprocesamiento de texto.

## Requisitos
- Python 3.x
- Bibliotecas: pandas, numpy, scikit-learn, nltk

## Instalación
1. Clona el repositorio:
   ```bash
   git clone <url-del-repositorio>
   cd <nombre-del-repositorio>
   ```
2. Instala las dependencias:
   ```bash
   pip install pandas numpy scikit-learn nltk
   ```
3. Ejecuta el script:
   ```bash
   python sentiment_analysis.py
   ```

## Conjunto de Datos
El script incluye un conjunto de datos de muestra pequeño para demostración. Para usar un conjunto de datos más grande, reemplaza el diccionario `data` en `sentiment_analysis.py` con tu propio conjunto de datos (por ejemplo, Sentiment140 o reseñas de IMDb).

## Estructura del Proyecto
- `sentiment_analysis.py`: Script principal que contiene el modelo de análisis de sentimientos.
- `README.md`: Este archivo.

## Uso
El script realiza los siguientes pasos:
1. Carga y preprocesa los datos de las reseñas.
2. Entrena un modelo de regresión logística utilizando características TF-IDF.
3. Evalúa el modelo y predice sentimientos para nuevas reseñas.

## Flujo de Trabajo en GitHub
1. Se creó una nueva rama: `feature/sentiment-model`
2. Commits:
   - Configuración inicial del proyecto con preprocesamiento básico
   - Agregado entrenamiento y evaluación del modelo
   - Agregado README y refinamientos finales

## Licencia
Licencia MIT