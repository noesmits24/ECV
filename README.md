
1. **Introducción**: Explicación breve de lo que hace la aplicación.
2. **Características**: Las funcionalidades clave que incluye la aplicación.
3. **Requisitos**: Las dependencias que necesita la aplicación.
4. **Instalación**: Instrucciones para clonar, instalar dependencias y ejecutar la aplicación.
5. **Uso**: Descripción de cómo usar la aplicación (tanto el dashboard como la predicción).
6. **Modelo de Machine Learning**: Explicación básica del modelo de predicción utilizado.
7. **Contribuciones y Licencia**: Para aquellos que quieran colaborar o usar el código en sus propios proyectos.


# Predicción de Enfermedad Coronaria

Esta aplicación web predice el riesgo de desarrollar una enfermedad coronaria en los próximos diez años. Utiliza un modelo de **Random Forest** entrenado con los datos del **Estudio del Corazón de Framingham**, e incluye una interfaz interactiva creada con **Streamlit** para explorar la distribución de varios factores de riesgo y predecir el riesgo basado en información personalizada.

## Características

- **Visualización interactiva**: Explora la distribución de la edad, el género, y los factores de riesgo clave como la glucosa, el colesterol, la presión arterial, etc.
- **Predicción de riesgo**: Introduce datos personales (presión arterial, nivel de glucosa, edad, entre otros) para obtener una predicción personalizada sobre el riesgo de enfermedad coronaria a diez años.
- **Gráficos**: Genera gráficos de distribución usando **Matplotlib** y gráficos interactivos con **Plotly**.

## Requisitos

Asegúrate de tener instaladas las siguientes dependencias para ejecutar la aplicación:

```txt
streamlit==1.27.1
pandas==2.0.3
numpy==1.24.3
matplotlib==3.7.2
plotly==5.9.0
scikit-learn==1.3.0
