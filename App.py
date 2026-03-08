import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#EJECUTAR STREAMLIT
#STREAMLIT RUN APP.PY

#CLASE
#############################
class DataAnalyzer:
    def __init__(self, df):
        self.df = df

    def clasificar_variables(self):
        numericas = self.df.select_dtypes(include=np.number).columns.tolist()
        categoricas = self.df.select_dtypes(include="object").columns.tolist()
        return numericas, categoricas

    def estadisticas(self):
        return self.df.describe()

    def valores_faltantes(self):
        return self.df.isnull().sum()

    def media(self, col):
        return self.df[col].mean()

    def mediana(self, col):
        return self.df[col].median()

    def moda(self, col):
        return self.df[col].mode()[0]

#STREAMLIT
#############################
st.title("TRABAJO FINAL - ESP. PYTHON DMC")

st.sidebar.image("Logo_DMC.png")
st.sidebar.title("Módulos")
pagina = st.sidebar.selectbox("Elige el módulo a visualizar:", ["Home","Carga dataset","Análisis exploratorio"])

#HOME
#############################
if pagina == "Home":
    st.header("Proyecto Aplicado Python: Análisis Churn")
    st.write("Objetivo del proyecto: Analizar los datos para comprender las causas asociadas a la fuga de clientes")
    st.write("Elaborado por Rodrigo Fernandez Arce")    
    st.write("Especiaización en Python para analítica")
    st.write("Año: 2026")
    st.write("Tecnologías: Python, Streamlit")

#CARGA DEL DATASET
#############################
elif pagina == "Carga dataset":
    
    st.header("Carga del Dataset")
    archivo = st.file_uploader("Sube el archivo", type=["csv"])

    if archivo is not None:
        df = pd.read_csv(archivo)
        st.session_state["df"] = df
        st.success("Archivo cargado correctamente")

        st.subheader("Vista previa del dataset")
        st.dataframe(df.head())

        filas, columnas = df.shape
        st.write(f"Filas: {filas}")
        st.write(f"Columnas: {columnas}")

    else:
        st.warning("Favor de cargar un archivo")

#ANÁLISIS EXPLORATORIO DE DATOS
#############################
elif pagina == "Análisis exploratorio":
    if "df" not in st.session_state:
        st.warning("Primero debes cargar el dataset")

    else:
        df = st.session_state["df"]
        analyzer = DataAnalyzer(df)

        st.header("Análisis exploratorio de datos")
        tabs = st.tabs([
            "Info dataset",
            "Clasificación variables",
            "Estadística",
            "Valores faltantes",
            "Distribución variables númericas",
            "Variables categóricas",
            "Numérico vs Categórico",
            "Categórico vs Categórico",
            "Análisis dinámico",
            "Conclusiones"
        ])
    
#ITEM 1
#############################
        with tabs[0]:
            st.subheader("Información general del dataset")
            st.write("Tipos de datos")
            st.write(df.dtypes)
            st.write("Valores nulos")
            st.write(df.isnull().sum())

#ITEM 2
#############################
        with tabs[1]:
            st.subheader("Clasificación de variables")
            numericas, categoricas = analyzer.clasificar_variables()

            col1, col2 = st.columns(2)
            with col1:
                st.write("Variables numéricas")
                st.write(numericas)
                st.write(f"Total: {len(numericas)}")
            with col2:
                st.write("Variables categóricas")
                st.write(categoricas)
                st.write(f"Total: {len(categoricas)}")

#ITEM 3
#############################
        with tabs[2]:
            st.subheader("Estadísticas descriptivas")
            st.dataframe(analyzer.estadisticas())

#ITEM 4
#############################
        with tabs[3]:
            st.subheader("Análisis de valores faltantes")
            faltantes = analyzer.valores_faltantes()
 
            st.write(faltantes)
            fig, ax = plt.subplots()
            faltantes.plot(kind="bar", ax=ax)
            st.pyplot(fig)

#ITEM 5
#############################
        with tabs[4]:
            st.subheader("Distribución de variables numéricas")
            numericas, _ = analyzer.clasificar_variables()
            variable = st.selectbox("Selecciona una variable", numericas)
            fig, ax = plt.subplots()
            sns.histplot(df[variable], kde=True, ax=ax)
            st.pyplot(fig)
 
            st.write(f"Media: {analyzer.media(variable):.2f}")
            st.write(f"Mediana: {analyzer.mediana(variable):.2f}")

#ITEM 6
#############################
        with tabs[5]:
            st.subheader("Análisis de variables categóricas")
            _, categoricas = analyzer.clasificar_variables()
            variable = st.selectbox("Selecciona variable categórica", categoricas)
            conteo = df[variable].value_counts()

            st.write(conteo)
            fig, ax = plt.subplots()
            sns.countplot(data=df, x=variable, ax=ax)
            plt.xticks(rotation=45)
            st.pyplot(fig)

#ITEM 7
#############################
        with tabs[6]:
            st.subheader("Análisis numérico vs categórico")
            numericas, categoricas = analyzer.clasificar_variables()
            num = st.selectbox("Variable numérica", numericas)
            cat = st.selectbox("Variable categórica", categoricas)
            fig, ax = plt.subplots()
            sns.boxplot(data=df, x=cat, y=num, ax=ax)
            st.pyplot(fig)

#ITEM 8
#############################
        with tabs[7]:
            st.subheader("Análisis categórico vs categórico")
            _, categoricas = analyzer.clasificar_variables()
            col1 = st.selectbox("Primera variable", categoricas)
            col2 = st.selectbox("Segunda variable", categoricas)
            tabla = pd.crosstab(df[col1], df[col2])

            st.write(tabla)
            fig, ax = plt.subplots()
            sns.heatmap(tabla, annot=True, fmt="d", cmap="Blues", ax=ax)
            st.pyplot(fig)

#ITEM 9
#############################
        with tabs[8]:
            st.subheader("Análisis dinámico")
            columnas = st.multiselect("Selecciona columnas", df.columns)

            if columnas:
                st.dataframe(df[columnas].head())
            rango = st.slider("Selecciona rango de filas", 0, len(df), (0, 100))
            st.dataframe(df.iloc[rango[0]:rango[1]])


#ITEM 10
#############################
        with tabs[9]:
            st.subheader("Hallazgos clave:")
            st.write("""
            1. Los clientes con contratos mensuales presentan mayor tasa de churn.
            2. Los clientes con menor tenure tienden a abandonar el servicio más frecuentemente.
            3. Los cargos mensuales altos muestran mayor asociación con churn.
            4. Algunos servicios como fibra óptica presentan mayor tasa de cancelación.
            5. Los clientes con contratos de largo plazo muestran mayor retención.
            """)

#FIN