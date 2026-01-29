import nbformat as nbf
import os

# Crear un nuevo notebook
nb = nbf.v4.new_notebook()
cells = []

# --- PHASE 1: BUSINESS UNDERSTANDING ---
cells.append(nbf.v4.new_markdown_cell("""# Analisis Exploratorio de Datos (EDA) - Students Social Media Addiction
## Metodologia CRISP-DM (Version Extendida)

Este notebook documenta un analisis profundo sobre la adiccion a las redes sociales en estudiantes. El estudio sigue las fases de la metodologia **CRISP-DM** para asegurar un enfoque estructurado y orientado a resultados.

---

## 1. Entendimiento del Negocio

### 1.1 Contexto
En la ultima decada, las redes sociales han pasado de ser herramientas de comunicacion a ecosistemas complejos diseñados bajo la "economia de la atencion". Para los estudiantes, esto significa una exposicion constante a estimulos que pueden alterar habitos de sueño, rendimiento academico y salud mental.

### 1.2 Objetivos de Investigacion
- **Identificar Patrones**: Determinar que plataformas dominan el mercado estudiantil y como varia el consumo por demografia.
- **Evaluar Impacto**: Cuantificar la relacion entre el tiempo de uso y el bienestar (salud mental y sueño).
- **Analizar Riesgos**: Detectar si existe una correlacion negativa entre el puntaje de adiccion y el rendimiento academico.
- **Factores Sociales**: Explorar como el estado sentimental y los conflictos influyen en el comportamiento digital.

### 1.3 Criterios de Exito
- Identificacion clara de las plataformas de mayor riesgo y perfiles de usuario vulnerables.
- Visualizacion de la brecha de rendimiento academico y salud mental segun niveles de adiccion.
- Proveer una base solida para que los tomadores de decisiones puedan diseñar estrategias de intervencion."""))

# --- PHASE 2: DATA UNDERSTANDING ---
cells.append(nbf.v4.new_markdown_cell("""## 2. Entendimiento de los Datos

### 2.1 Diccionario de Datos
Variables detectadas en el dataset:

| Columna Original | Descripcion |
|------------------|-------------|
| **Student_ID** | Identificador unico del estudiante. |
| **Age** | Edad del estudiante. |
| **Gender** | Genero del estudiante. |
| **Academic_Level** | Nivel academico (Undergraduate, High School, etc.). |
| **Country** | Pais de origen. |
| **Avg_Daily_Usage_Hours** | Horas promedio diarias en redes sociales. |
| **Most_Used_Platform** | Plataforma mas utilizada. |
| **Affects_Academic_Performance** | Si afecta el rendimiento (Yes/No). |
| **Sleep_Hours_Per_Night** | Horas de sueño por noche. |
| **Mental_Health_Score** | Puntaje de salud mental. |
| **Relationship_Status** | Estado sentimental (Single, In Relationship, etc.). |
| **Conflicts_Over_Social_Media** | Escala de conflictos por uso de redes sociales. |
| **Addicted_Score** | Puntaje de adiccion (final). |"""))

cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
import warnings

# Configuraciones globales
warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11

# Carga de datos
df = pd.read_csv('Students Social Media Addiction.csv')
print(f"Dataset cargado con {df.shape[0]} registros y {df.shape[1]} columnas.")"""))

cells.append(nbf.v4.new_markdown_cell("""### 2.2 Exploracion Inicial e Insights Rapidos"""))

cells.append(nbf.v4.new_code_cell("""# Estadisticas descriptivas generales
display(df.describe().T)

print("\\n--- Distribucion Categorica ---")
for col in ['Gender', 'Most_Used_Platform', 'Academic_Level', 'Relationship_Status']:
    if col in df.columns:
        print(f"\\nRecuento de {col}:")
        display(df[col].value_counts())"""))

# --- PHASE 3: DATA PREPARATION ---
cells.append(nbf.v4.new_markdown_cell("""## 3. Preparacion de los Datos"""))

cells.append(nbf.v4.new_code_cell("""# 3.1 Limpieza
df.columns = [c.strip() for c in df.columns]
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# 3.2 Manejo de nulos
df = df.dropna()

print("Datos limpios y listos para el analisis.")"""))

# --- PHASE 4: MODELADO (VISUALIZACIONES) ---
cells.append(nbf.v4.new_markdown_cell("""---
## 4. Modelado (Analisis de Visualizaciones)
"""))

# 4.1 Dominancia de Plataformas
cells.append(nbf.v4.new_code_cell("""# Grafico 1: Ecosistema de Plataformas
plt.figure(figsize=(10, 6))
sns.countplot(data=df, y='Most_Used_Platform', order=df['Most_Used_Platform'].value_counts().index, palette='viridis')
plt.title('Dominancia de Plataformas en la Poblacion Estudiantil')
plt.xlabel('Numero de Estudiantes')
plt.show()
print("Proposito: Identificar que plataformas concentran la mayor atencion de los estudiantes.")"""))

# 4.2 Tiempo vs Adiccion
cells.append(nbf.v4.new_code_cell("""# Grafico 2: Relacion Tiempo vs Puntaje de Adiccion
plt.figure(figsize=(10, 6))
sns.regplot(data=df, x='Avg_Daily_Usage_Hours', y='Addicted_Score', scatter_kws={'alpha':0.4}, line_kws={'color':'red'})
plt.title('Correlacion entre Horas de Uso y Puntaje de Adiccion')
plt.xlabel('Horas Diarias')
plt.ylabel('Puntaje de Adiccion')
plt.show()
print("Proposito: Validar si existe una tendencia lineal clara entre el tiempo invertido y el sentimiento de adiccion.")"""))

# 4.3 Impacto en Rendimiento
cells.append(nbf.v4.new_code_cell("""# Grafico 3: Tiempo de Uso vs Afectacion Academica
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Affects_Academic_Performance', y='Avg_Daily_Usage_Hours', palette='Set2')
plt.title('Distribucion de Tiempo de Uso segun Afectacion Academica')
plt.xlabel('¿Percibe afectacion academica?')
plt.ylabel('Horas diarias')
plt.show()
print("Proposito: Comparar si los estudiantes que perciben una afectacion realmente promedian mas horas de uso.")"""))

# 4.4 Estado Sentimental vs Uso
cells.append(nbf.v4.new_code_cell("""# Grafico 4: Tiempo de Uso segun Estado Sentimental
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='Relationship_Status', y='Avg_Daily_Usage_Hours', palette='coolwarm')
plt.title('Habitos de Consumo Digital segun Situacion Sentimental')
plt.xlabel('Estado Sentimental')
plt.ylabel('Horas diarias')
plt.show()
print("Proposito: Explorar si la vida social presencial (relaciones) influye en el tiempo dedicado a redes sociales.")"""))

# 4.5 Conflictos vs Salud Mental
cells.append(nbf.v4.new_code_cell("""# Grafico 5: Impacto de los Conflictos en la Salud Mental
plt.figure(figsize=(12, 6))
sns.barplot(data=df, x='Conflicts_Over_Social_Media', y='Mental_Health_Score', ci=None, palette='Reds_d')
plt.title('Nivel de Conflictos vs Puntaje de Salud Mental')
plt.xlabel('Escala de Conflictos (Mayor = Mas conflictos)')
plt.ylabel('Puntaje Promedio de Salud Mental')
plt.show()
print("Proposito: Determinar si el aumento en la intensidad de conflictos digitales impacta negativamente el bienestar percibido.")"""))

# 4.6 Uso por Nivel Academico
cells.append(nbf.v4.new_code_cell("""# Grafico 6: Adiccion segun Nivel Academico
plt.figure(figsize=(12, 6))
sns.violinplot(data=df, x='Academic_Level', y='Addicted_Score', palette='pastel')
plt.title('Distribucion de Adiccion por Nivel de Estudios')
plt.xlabel('Nivel Academico')
plt.ylabel('Puntaje de Adiccion')
plt.show()
print("Proposito: Observar si la 'madurez academica' o la carga de estudio influye en la adiccion.")"""))

# 4.7 Heatmap Final
cells.append(nbf.v4.new_code_cell("""# Grafico 7: Matriz de Correlacion Global
numeric_cols = df.select_dtypes(include=[np.number]).columns
corr = df[numeric_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
plt.figure(figsize=(12, 10))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Panorama Global de Correlaciones de Bienestar')
plt.show()
print("Proposito: Resumir estadisticamente todas las relaciones clave del dataset.")"""))

# --- EVALUATION ---
cells.append(nbf.v4.new_markdown_cell("""---
## 5. Evaluacion

**[ESPACIO PARA ANALISIS DE USUARIO]**

Basandose en las 7 visualizaciones presentadas:
1. ¿Cuales son los 3 insights mas criticos para una institucion educativa?
2. ¿Hay alguna plataforma que deba ser el foco de campañas de prevencion?
3. ¿Que papel juega la salud mental en este ecosistema? """))

# Guardar
nb['cells'] = cells
with open('EDA_Students_Social_Media_Full_CRISPDM.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Notebook avanzado generado exitosamente.")
