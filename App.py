#import libraries
import streamlit as st  # for creating web apps
import numpy as np  # for numerical computing
import pandas as pd  # for dataframe manipulation
from matplotlib import pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px  # for graphs and data visualization
import pickle
import gdown  # Importing gdown to download from Google Drive

# CSS Styling
with open('style.css') as f:
    css = f.read()
st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

# Download the model from Google Drive
@st.cache_data
def download_model():
    # File ID from your Google Drive link
    file_id = '13jKzu4qLz12rVKho60oi0kAb5il4JN6D'
    
    # Construct the download URL
    download_url = f'https://drive.google.com/uc?id={file_id}'
    
    # Specify the output file name
    output = 'rf_hyper_model.pkl'
    
    # Download the file from Google Drive
    gdown.download(download_url, output, quiet=False)
    return output

# Load model and scaler
@st.cache_data
def load_model_and_scaler():
    # Download model from Google Drive
    model_path = download_model()

    # Load the model and scaler
    with open(model_path, 'rb') as model_file, open('scaler.pkl', 'rb') as scaler_file:
        model = pickle.load(model_file)
        scaler = pickle.load(scaler_file)
    
    return model, scaler

# Loading the dataset
@st.cache_resource
def load_data():
    return pd.read_csv('framingham_clean.csv')

df = load_data()

# Plot functions using Matplotlib
def plot_histogram(df, column, ax, title=None):
    ax.hist(df[column], bins=20, color='blue', alpha=0.7)
    ax.set_title(f'Distribución de {column}' if not title else title, fontsize=14)
    ax.set_xlabel(column, fontsize=12)
    ax.set_ylabel('Frecuencia', fontsize=12)

def plot_count(df, column, ax, title=None):
    counts = df[column].value_counts()
    ax.bar(counts.index, counts.values, color='blue', alpha=0.7)
    ax.set_title(f'Distribución de {column}' if not title else title, fontsize=14)
    ax.set_xlabel(column, fontsize=12)
    ax.set_ylabel('Cantidad', fontsize=12)

# Home Page
def show_home_page():
    st.markdown('<h1 class="my-title ">Estudio del Corazón de Framingham</h1>', unsafe_allow_html=True)
    st.markdown('<h3 class="sub-header">1- Distribución de Edad y Género en el Conjunto de Datos.</h3>', unsafe_allow_html=True)
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    most_frequent_age = df['age'].mode().values[0]
    
    # Histograma de edades
    plot_histogram(df, 'age', ax1, title='Distribución de la Edad (Edad Más Frecuente)')
    ax1.annotate(f'Edad Más Frecuente: {most_frequent_age}', xy=(most_frequent_age, 0), xytext=(most_frequent_age, 50),
                 arrowprops=dict(arrowstyle='->', lw=1.5, color='red'), color='red')

    # Gráfico de barras para género
    plot_count(df, 'gender', ax2, title='Distribución de Género')
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(['Mujer', 'Hombre'])
    
    st.pyplot(fig)

    with st.expander("2. Distribución de Factores de Riesgo Cardiovascular Claves con Etiquetas de Asimetría"):
        # Crear gráficos para múltiples variables
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        variables = ['glucose', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate']
        
        for i, var in enumerate(variables):
            row, col = i // 3, i % 3
            ax = axes[row, col]
            plot_histogram(df, var, ax, title=f'Distribución de {var}')
        
        st.pyplot(fig)

    st.markdown('<h3 class="sub-header">3. Diabetes y Enfermedad Cardiovascular por Grupo de Edad y Género</h2>', unsafe_allow_html=True)

    # Subplot para la visualización de diabetes y ECV por edad y género
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Primera subgráfica: Diabetes por Grupo de Edad
    plot_count(df, 'age_groups', axes[0], title='Diabetes por Grupo de Edad')
    axes[0].legend(['Negativo', 'Positivo'], title='Diabetes')
    
    # Segunda subgráfica: ECV por Género
    plot_count(df, 'gender', axes[1], title='Enfermedad Cardiovascular por Género')
    axes[1].legend(['Negativo', 'Positivo'], title='Estado ECV')
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(['Mujer', 'Hombre'])

    st.pyplot(fig)

# Prediction Page
def show_prediction_page():
    st.markdown('<h1 class="my-title">Predicción de Enfermedad Cardiovascular</h1>', unsafe_allow_html=True)
    st.markdown('<h3 class="sub-header">1- Introduzca sus datos para predecir el riesgo de desarrollar enfermedad cardiovascular en los próximos diez años. </h3>', unsafe_allow_html=True)

    model, scaler = load_model_and_scaler()
    
    with st.sidebar:
        st.markdown('<h2 style="color: orange; text-align: center;">Ingrese sus detalles:</h2>', unsafe_allow_html=True)
        sysBP = st.number_input('Presión Arterial Sistólica', 80, 200, 120)
        glucose = st.number_input('Nivel de Glucosa', 40, 400, 100)
        age = st.number_input('Edad', 30, 80, 50)
        cigsPerDay = st.number_input('Cigarrillos por Día', 0, 60, 10)
        totChol = st.number_input('Colesterol Total', 100, 600, 250)
        diaBP = st.number_input('Presión Arterial Diastólica', 60, 140, 80)

    st.markdown('<h4 style="color: orange; text-align: center;">Detalles Adicionales:</h4>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        prevalentHyp_value = st.selectbox('Hipertensión Prevalente', ['No', 'Sí'])
        BPMeds_value = st.selectbox('Medicamentos para la Presión Arterial', ['No', 'Sí'])
    with col2:
        diabetes_value = st.selectbox('Diabetes', ['No', 'Sí'])
        gender_value = st.selectbox('Género', ['Mujer', 'Hombre'])

    if st.button('Predecir Riesgo'):
        user_data = [[sysBP, glucose, age, cigsPerDay, totChol, diaBP,
                    1 if prevalentHyp_value == 'Sí' else 0,
                    1 if BPMeds_value == 'Sí' else 0,
                    1 if diabetes_value == 'Sí' else 0,
                    1 if gender_value == 'Hombre' else 0]]
        user_data_scaled = scaler.transform(user_data)
        prediction = model.predict(user_data_scaled)
        probability = model.predict_proba(user_data_scaled)[0][1]

        alert_type = "Alto Riesgo de ECV ⚠️" if prediction[0] == 1 else "Bajo Riesgo de ECV ✅"
        alert_color = "#ffa1a1" if prediction[0] == 1 else "#a1ffad"
        st.markdown(f"""
            <div style="background-color: {alert_color}; padding: 10px; border-radius: 5px; text-align: center;">
                {alert_type}
            </div>
        """, unsafe_allow_html=True)

# Main Function
def main():
    with st.sidebar:
        st.markdown('<h1 class="sidebar-title">Enfermedad Cardiovascular</h1>', unsafe_allow_html=True)
        selected = st.sidebar.selectbox("Selecciona una opción", ["Dashboard", "Aplicación"])
    
    if selected == "Dashboard":
        show_home_page()
    elif selected == "Aplicación":
        show_prediction_page()

if __name__ == "__main__":
    main()

st.sidebar.markdown("<h4 style='color: blue; font-size: 16px;'>Desarrollado por DSO Immune Grp1 🌐</h4>", unsafe_allow_html=True)
