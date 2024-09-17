#import libraries
import streamlit as st  # for creating web apps
from streamlit_option_menu import option_menu
import numpy as np  # for numerical computing
import pandas as pd # for dataframe and manipulation
import seaborn as sns # for graphs and data visualization
from matplotlib import pyplot as plt 
import plotly.graph_objects as go 
from plotly.subplots import make_subplots
import plotly.express as px # for graphs and data visualization
sns.set() # setting seaborn as default for plots
import pickle

# CSS Styling

# Load the contents of CSS file
with open('style.css') as f:
    css = f.read()

# Use the st.markdown function to apply the CSS to Streamlit app
st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

# Loading the dataset
@st.cache_resource # Cache the data loading step to enhance performance
def load_data():
    return pd.read_csv('framingham_clean.csv')

df = load_data()

def show_home_page():
    st.markdown('<h1 class="my-title ">Estudio del Corazón de Framingham</h1>', unsafe_allow_html=True)
    st.markdown('<h3 class="sub-header">1- Distribución de Edad y Género en el Conjunto de Datos.</h3>', unsafe_allow_html=True)
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    most_frequent_age = df['age'].mode().values[0]
    sns.histplot(df['age'], bins=20, kde=True, ax=ax1)
    ax1.set_title('Distribución de la Edad (Edad Más Frecuente)')
    ax1.set(xlabel='Edad', ylabel='Frecuencia')
    ax1.annotate(f'Edad Más Frecuente: {most_frequent_age}', xy=(most_frequent_age, 0), xytext=(most_frequent_age, 50),
                 arrowprops=dict(arrowstyle='->', lw=1.5, color='red'), color='red')

    # Gender distribution plot
    sns.countplot(data=df, x='gender', ax=ax2)
    ax2.set_title('Distribución de Género')
    ax2.set_xlabel('Género')
    ax2.set_ylabel('Cantidad')
    total_count = len(df)
    for p in ax2.patches:
        percentage = f'{100 * p.get_height() / total_count:.1f}%'
        ax2.annotate(percentage, (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom')
    ax2.set_xticklabels(['Mujer', 'Hombre'])
    st.pyplot(fig)

    with st.expander("2. Distribución de Factores de Riesgo Cardiovascular Claves con Etiquetas de Asimetría"):
        # Define colors for each variable
        colors = ['orange', 'green', 'red', 'purple', 'blue', 'grey']

        # Create a figure to contain all the subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.subplots_adjust(wspace=0.3, hspace=0.3)

        variables = ['glucose', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate']

        # Define skewness labels based on your criteria
        def get_skew_label(skewness):
            if skewness < -1 or skewness > 1:
                return 'Altamente Sesgado'
            elif (-1 <= skewness <= -0.5) or (0.5 <= skewness <= 1):
                return 'Moderadamente Sesgado'
            else:
                return 'Aproximadamente Simétrico'

        for i, var in enumerate(variables):
            row, col = i // 3, i % 3
            ax = axes[row, col]

            # Calculate skewness
            skewness = df[var].skew()
            skew_label = get_skew_label(skewness)

            sns.histplot(df[var], color=colors[i], kde=True, ax=ax)
            ax.set_title(f'Distribución de {var}\nAsimetría: {skewness:.2f} ({skew_label})')

        # Display the entire figure in Streamlit
        st.pyplot(fig)

    st.markdown('<h3 class="sub-header">3. Diabetes y Enfermedad Coronaria por Grupo de Edad y Género</h2>', unsafe_allow_html=True)

    # Create a figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # First subplot: Diabetes by Age Group
    plt.sca(axes[0])
    ax1 = sns.countplot(x='age_groups', hue='diabetes', data=df, palette='rainbow')
    plt.xlabel('Grupo de Edad')
    plt.ylabel('Número de Pacientes')
    plt.legend(title='Diabetes', labels=['Negativo', 'Positivo'])
    plt.title('Diabetes por Grupo de Edad')
    total_count1 = len(df)
    for p in ax1.patches:
        height = p.get_height()
        percentage = height / total_count1 * 100
        ax1.text(p.get_x() + p.get_width() / 2, height + 5, f'{percentage:.1f}%', ha='center')

    # Second subplot: Coronary Heart Disease by Gender
    plt.sca(axes[1])
    sns.set_style("whitegrid")
    ax2 = sns.countplot(x='gender', hue='TenYearCHD', data=df, palette='Paired')
    plt.xlabel('Género')
    plt.xticks(ticks=[0, 1], labels=['Mujer', 'Hombre'])
    plt.ylabel('Número de Pacientes')
    plt.legend(['Negativo', 'Positivo'], title='Estado CHD')
    plt.title('Enfermedad Coronaria por Género')
    total_count2 = len(df)
    for p in ax2.patches:
        height = p.get_height()
        percentage = height / total_count2 * 100
        ax2.text(p.get_x() + p.get_width() / 2, height + 5, f'{percentage:.1f}%', ha='center')

    sns.despine(left=True, ax=axes[1])
    axes[1].set_axisbelow(True)

    # Display the subplots in Streamlit
    st.pyplot(fig)

def show_prediction_page():
    st.markdown('<h1 class="my-title">Predicción de Enfermedad Coronaria</h1>', unsafe_allow_html=True)
    st.markdown('<h3 class="sub-header">1- Introduzca sus datos para predecir el riesgo de desarrollar enfermedad coronaria en los próximos diez años. </h3>', unsafe_allow_html=True)

    @st.cache_data
    def load_model_and_scaler():
        with open('rf_hyper_model.pkl', 'rb') as model_file, open('scaler.pkl', 'rb') as scaler_file:
            model = pickle.load(model_file)
            scaler = pickle.load(scaler_file)
        from copy import deepcopy
        return deepcopy(model), deepcopy(scaler)

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

        # New emoticons
        alert_type = "Alto Riesgo de CHD ⚠️" if prediction[0] == 1 else "Bajo Riesgo de CHD ✅"
        alert_color = "#ffa1a1" if prediction[0] == 1 else "#a1ffad"
        st.markdown(f"""
            <div style="background-color: {alert_color}; padding: 10px; border-radius: 5px; text-align: center;">
                {alert_type}
            </div>
        """, unsafe_allow_html=True)

def main():
    with st.sidebar:
        st.markdown('<h1 class="sidebar-title">Enfermedad Coronaria</h1>', unsafe_allow_html=True)
        selected = option_menu(None, ["Dashboard CHD", "Predicción CHD"], icons=["bar-chart", "robot"], default_index=0)
       
    if selected == "Dashboard CHD":
        show_home_page()
    elif selected == "Predicción CHD":
        show_prediction_page()

if __name__ == "__main__":
    main()

st.sidebar.markdown("<h4 style='color: blue; font-size: 16px;'>Desarrollado por DSO Immune 🌐</h4>", unsafe_allow_html=True)
