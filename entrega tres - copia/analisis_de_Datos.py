# Libreria que permite manipular el data_frame (df)
import pandas as pd # type: ignore
# Clase que permite detectar el lenguaje del taxto, y poder leer correctamente el archivo
from charamel import Detector # type: ignore
# Libreria que permite hacer las gradicas 
import matplotlib.pyplot as plt # type: ignore
# Libreria que permite realizar el grafico interactivo
import seaborn as sns # type: ignore
# Permite hacer regresion lineal
from sklearn.linear_model import LinearRegression # type: ignore

#  ---------------------------------- Paso 1: Exploración y Limpieza de Datos (Revisión inicial del dataset con Pandas, Manejo de valores nulos, duplicados o inconsistencias. -----------------------------------
# (Revisión inicial del dataset COVID-19.csv) indica que separa los encabezados por ; // Iindica la codificacion del texto del archivo
df = pd.read_csv('COVID-19.csv', sep=";", encoding="iso-8859-1") 
print(df) # Imprime el data frame
print(type(df)) # Imprime el tipo de la variable data frame

# Reemplazar espacios por guiones bajos
df.columns = df.columns.str.replace(' ', '_')  
print(df.columns) # Imprime los encabezados de las columnas

# valores duplicados
df.drop_duplicates(inplace=True)

# Suma cuantos valores nulos hay
print(df.isnull().sum()) 

# Cambia el nombre de las columnas para evitar errores mas adelante 
df.rename(columns={
    'Fecha_de_notificaciÃ³n': 'Fecha_notificacion',
    'atenciÃ³n': 'Atencion',
    'Ciudad_de_ubicaciÃ³n': 'Ciudad_ubicacion',
    'PaÃ­s_de_procedencia': 'Pais_procedencia'
}, inplace=True)
print(df.columns) # Imprime los encabezados de las columnas

# valores nulos
print(df[['Departamento_o_Distrito_', 'Sexo', 'Edad', 'Estado', 'Fecha_notificacion', 'Fecha_diagnostico']].isnull().sum()) # Cuenta cada valor nulo de cada columna nombrada
df.dropna(subset=['Departamento_o_Distrito_', 'Sexo', 'Edad', 'Estado', 'Fecha_notificacion', 'Fecha_diagnostico'], inplace=True) # Elimina la fila con vlaores nulos
print(df[['Departamento_o_Distrito_', 'Sexo', 'Edad', 'Estado', 'Fecha_notificacion', 'Fecha_diagnostico']].isnull().sum()) # Vuelve a contar la cantidad de valores nulos por columna, para verificar si los elimino

# Elimina valores duplicados
df.drop_duplicates(inplace=True)

# Imprime las columnas a utilizar
print(df['Departamento_o_Distrito_'])
print(df['Sexo'])
print(df['Edad'])
print(df['Estado'])
print(df['Fecha_notificacion'])
print(df['Fecha_diagnostico'])

#  -------------------------------------- Paso 2: 3.3 Transformación de Datos (Creación de nuevas variables o columnas, Conversión de formatos y normalización de datos) ----------------------------------------
# Revisar los tipos de datos para parserlos
print(df.info())

# Convertir la columna 'Edad' a tipo numérico, forzando errores a NaN
df['Edad'] = pd.to_numeric(df['Edad'], errors='coerce')
# Convertir las fechas a tipo date
fechas = ["Fecha_notificacion", "Fecha_diagnostico"] # Nombra las fechas en una sola variable
df[fechas] = df[fechas].apply(pd.to_datetime, errors='coerce') # Convirte la variable con las fechas a tipo date
print(df.dtypes) # Imprime el tipo de los datos de cada columna

# SERIES

# Serio de casos de recuperados
df["Recuperado"] = df["Estado"].apply(lambda x: 0 if str(x).lower() == "fallecido" else 1)
print(df[['Estado', 'Recuperado']]) # Imprime la serie Estado y la serie recien creada Recuperado
print(df.loc[2])

# Serie: Caso importado 
# Toma la serie de fecha notificacion, y empieza ocmparar para sacar el mes aproximado de contagio
df['Mes_anio'] = df['Fecha_notificacion'].dt.to_period('M')
print(df[['ID_de_caso', 'Mes_anio']]) # Imprime el id del caso y la serie recien creada
print(df.loc[2]) # Imprime una linea especifica

# Toma la fecha de diagnostico del caso y le resta la fecha de notificaicon y a ese resultado lo pasa a dias
df["Retraso_notif_diagnostico"] = (df["Fecha_diagnostico"] - df["Fecha_notificacion"]).dt.days 
print(df[['ID_de_caso', 'Retraso_notif_diagnostico']]) # Imprime el id del caso y la serie recien creada

# Serie Categoría por edad
# Crea una condicion la cual resive como parametro edad, la cual copara en tres ocaciones asignandole una etiqueta a cada comparacion
def clasificar_edad(edad):
    if edad < 18:
        return "Niño"
    elif edad < 60:
        return "Adulto"
    else:
        return "Adulto Mayor"
# Crea un nueva serie, toma la edad y le aplica la funcion creada anteriormente, clasificando cada edad
df["Categoria_edad"] = df["Edad"].apply(clasificar_edad)
print(df[['ID_de_caso', 'Categoria_edad']])

#  ---------------------------------- Paso 3: Manejo de valores, valores nulos y valores duplicados -----------------------------------
# CALCULOS
# Agrupa los datos por el mes de reporte
casos_mensuales = df.groupby('Mes_anio').size()
print(casos_mensuales)

# Agrupa por el campo sexo y hace el conteo
Conteo_sexo = df.groupby("Sexo").count()
print(Conteo_sexo)

# Analisis de picos
picos = ['2020-03', '2020-04', '2020-05']
cantidades = [casos_mensuales.get(pico, 0) for pico in picos]

# Agrupa los datos por estado de paciente y uego los cuenta
Conteo_estado = df.groupby("Estado").count()
print(Conteo_estado)

# Cuenta cunados recuperdos
conteo_recuperados = df['Recuperado'].value_counts().sort_index()

# Graficas
# ---------- 4.1 Análisis Exploratorio de Datos EDA (Estadísticas descriptivas, Distribuciones, correlaciones, tendencias preliminares.) ----------
# Muestra la cantidad de casos por edad // Interactivo
plt.figure(figsize=(10, 5))
sns.histplot(df['Edad'].dropna(), kde=True)
plt.title('Distribución de Edad')
plt.xlabel('Edad')
plt.ylabel('Frecuencia')
plt.show()

# Visualizacio de tendencias de casos por mes: 
plt.figure(figsize=(12, 6))
casos_mensuales.plot(marker='o')
plt.title('Tendencia de Casos Mensuales')
plt.xlabel('Mes-Año')
plt.ylabel('Número de Casos')
plt.grid(True)
plt.show()

# Grafica los picos de casos en el mes
plt.figure(figsize=(8, 5))  # Tamaño del gráfico
plt.bar(picos, cantidades, color='skyblue')
plt.title('Casos en meses clave')
plt.xlabel('Mes')
plt.ylabel('Cantidad de casos')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Graficar un gráfico de barras horizontales (barh) usando la columna "ID_de_caso" de Conteo_estado
plt.figure(figsize=(8, 6))
Conteo_estado["ID_de_caso"].plot(kind='barh', color='coral')
plt.title("Conteo de casos por Estado")
plt.xlabel("Número de casos")
plt.ylabel("Estado")
plt.tight_layout()
plt.show()

# ----------  4.2. Análisis Estadístico (Aplicación de métodos estadísticos (si aplica: regresión, prueba de hipótesis, etc.), Resultados y su interpretación..) ----------
# Regesion linel
X = df[['Retraso_notif_diagnostico']].dropna()
Y = df.loc[X.index, 'Edad']
modelo = LinearRegression()
modelo.fit(X, Y)
Y_pred = modelo.predict(X)
# Graficar puntos y línea de regresión
plt.scatter(X, Y, color='blue', label='Datos')
plt.plot(X, Y_pred, color='red', linewidth=2, label='Regresión lineal')
plt.xlabel('Días entre notificación y diagnóstico')
plt.ylabel('Edad')
plt.title('Regresión lineal: Edad vs Días entre notificación y diagnóstico')
plt.legend()
plt.show()

# ---------- 4.3 Identificación de Insights (Principales hallazgos del análisis, Relevancia de estos hallazgos para el problema planteado.)
# Casos por sexo
plt.figure(figsize=(6, 6))
Conteo_sexo["ID_de_caso"].plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=['#ff9999', '#66b3ff'])
plt.title("Distribución por Sexo")
plt.ylabel("")
plt.show()
    
# Grafica de fallecidos y recuperados
plt.figure(figsize=(6, 4))
plt.bar(['Fallecido', 'Recuperado'], conteo_recuperados, color=['red', 'green'])
plt.title('Casos por estado de recuperación')
plt.ylabel('Cantidad de casos')
plt.xlabel('Estado')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
