# Plan Detallado — Reto 2: Clasificación Binaria con ANN (Hadronterapia)

**Curso:** IBIO-2340 Fundamentos de Machine Learning  
**Grupo:** 9 — Sección 2  
**Integrantes:** Santiago Casasbuenas, Juan Pablo Castro, Ana Cristina Rodríguez  

---

## 1. Resumen del Problema

Se ha desarrollado un proceso de bajo costo para generar haces de protones para hadronterapia, pero este proceso también produce rayos gamma residuales. Se necesita un modelo de **Red Neuronal Artificial (ANN)** que clasifique cada haz como **hadrón (clase 1)** o **gamma (clase 0)** a partir de 6 características medidas por un detector. Esto permite filtrar los haces antes de su transporte al gantry, garantizando que solo se irradie al paciente con hadrones (beneficio del pico de Bragg).

---

## 2. Datos Disponibles

| Archivo | Descripción | Filas | Columnas |
|---------|-------------|-------|----------|
| `hadron_train.csv` | Entrenamiento + validación | 10,700 | 6 features + clase |
| `hadron_test.csv` | Predicciones finales (sin etiquetas) | 2,676 | 6 features |
| `helpers - Copy.py` | Función `check_requirements()` para validar `predictions.csv` | — | — |

### Features

| Feature | Tipo | Rango aprox. | Descripción |
|---------|------|-------------|-------------|
| `fLength` | float64 | [4.28, 334.18] | Longitud mayor del elipsoide |
| `fWidth` | float64 | [0.00, 228.04] | Longitud menor del elipsoide |
| `fConc1` | float64 | [0.01, 0.89] | Concentración (relación del pixel más brillante) |
| `fM3Long` | float64 | [-331.78, 283.86] | Tercer momento a lo largo del eje mayor |
| `fAlpha` | float64 | [0.00, 90.00] | Ángulo del eje mayor respecto al vector al centro |
| `fDist` | float64 | [1.28, 450.95] | Distancia del origen al centro del elipsoide |

### Distribución de clases (entrenamiento)

- **Gamma (0):** 5,364 (50.1%)
- **Hadrón (1):** 5,336 (49.9%)
- Clases prácticamente balanceadas — no se necesitan técnicas de balanceo agresivas.

### Observaciones clave

- **No hay valores nulos** en ninguna feature.
- Ambos CSVs tienen una **columna índice sin nombre** como primera columna → cargar con `index_col=0`.
- Los rangos de las features son **muy heterogéneos**: fConc1 está en [0,1] mientras fDist llega a 450. Esto hace obligatoria la estandarización.
- fM3Long tiene **valores negativos** (hasta -331), lo cual descarta MinMaxScaler si se quiere preservar la distribución natural.

---

## 3. Rúbrica de Calificación (Análisis Detallado)

### 3.1 Informe (50% de la nota)

| Criterio | Peso | Qué evalúa | Estrategia |
|----------|------|------------|------------|
| Redacción, organización, buenas prácticas | 5% | Notebook limpio, código documentado, estructura clara | Secciones con títulos claros, código modular con funciones helper |
| Formulación matemática en **LaTeX** | 15% | Ecuaciones correctas y completas | Sección 2 dedicada con TODAS las ecuaciones en `$$...$$` |
| Profundidad en análisis del modelo | 15% | Comparación de modelos, curvas de aprendizaje | Tabla comparativa + gráficas de learning curves para cada modelo |
| Modelo final claro y justificado | 5% | Arquitectura e hiperparámetros explícitos | Sección dedicada con `model.summary()` y tabla de hiperparámetros |
| Análisis desde contexto del problema | 10% | Interpretación clínica de resultados | Discusión de FP/FN en contexto de hadronterapia |

**RESTRICCIÓN CRÍTICA:** Si las ecuaciones NO están en LaTeX → nota 1.0 automática.

### 3.2 Rendimiento del Modelo (50% de la nota)

- Se calcula el **recall** del modelo sobre `predictions.csv` (comparado contra etiquetas reales que tiene el profesor).
- Fórmula de calificación:

$$C = \frac{4.5}{\max - \min} \cdot (\text{Recall} - \min) + 0.5$$

- El grupo con **mayor recall** obtiene 5.0, el de **menor recall** obtiene 0.5.
- **RESTRICCIÓN CRÍTICA:** Recall < 0.7 → nota 1.0 automática.

### 3.3 Implicaciones estratégicas

1. **Recall es la métrica que importa para el 50% de la nota.** Toda decisión de diseño debe priorizar maximizar recall.
2. El recall se mide sobre la clase positiva (hadrón = 1): $\text{Recall} = \frac{TP}{TP + FN}$.
3. Optimizar recall implica minimizar falsos negativos (hadrones clasificados como gamma).
4. Podemos sacrificar algo de precision a cambio de mayor recall sin penalización en la nota.

---

## 4. Restricciones Técnicas

| Restricción | Detalle |
|-------------|---------|
| Framework | **Keras (TensorFlow)** únicamente para DL |
| Librerías permitidas | NumPy, Pandas, Matplotlib, Scikit-Learn, Scipy |
| Librerías prohibidas | Cualquier otra librería de DL/ML (PyTorch, XGBoost, LightGBM, scikeras, etc.) |
| Arquitectura | Desde cero — **NO modelos pre-entrenados, NO transfer learning** |
| Capa de entrada | Neuronas = número de features (6) |
| Capas ocultas | Mínimo 1 capa Dense con activación no lineal (ReLU) |
| Capa de salida | Clasificación binaria (sigmoid) |
| Loss | Apropiada para clasificación binaria (binary_crossentropy) |
| Optimizador | Obligatorio (es un hiperparámetro a elegir) |
| Métricas | Obligatorio incluir **matriz de confusión** y **recall** |
| Regularización permitida | Dropout, Early Stopping |
| Selección de modelo | GridSearch permitido |
| predictions.csv | 7 columnas (6 features + "prediction"), 2676 filas, sin índice |

---

## 5. Estructura del Notebook — 11 Secciones

### Sección 1: Encabezado y Explicación del Problema

**Tipo:** Markdown  
**Contenido:**
- Título del reto, integrantes, grupo, sección (seguir formato del Lab12)
- Explicación del problema en palabras propias (2-3 párrafos en español):
  - Qué es la hadronterapia y por qué es prometedora (pico de Bragg, mayor precisión)
  - El problema del nuevo generador de bajo costo (produce gamma residuales)
  - El objetivo de clasificación: distinguir hadrón vs gamma para filtrar antes del gantry
  - Importancia clínica: seguridad del paciente, eficacia del tratamiento

**Rubrica que cubre:** Redacción y organización (5%), Análisis contextual (10% parcial)

---

### Sección 2: Definición del Problema de Clasificación y Formulación Matemática

**Tipo:** Markdown con LaTeX  
**Esta sección vale 15% de la nota total (30% del informe). Es CRÍTICA.**

**Contenido detallado:**

#### 2.1 Definición formal del problema
- Conjunto de entrenamiento: $\{(\mathbf{x}^{(i)}, y^{(i)})\}_{i=1}^{m}$, con $\mathbf{x}^{(i)} \in \mathbb{R}^6$, $y^{(i)} \in \{0, 1\}$, $m = 10700$
- Features: $\mathbf{x} = (x_1, x_2, x_3, x_4, x_5, x_6)^T$ = (fLength, fWidth, fConc1, fM3Long, fAlpha, fDist)
- Objetivo: aprender $f: \mathbb{R}^6 \to \{0,1\}$

#### 2.2 Neurona artificial
- Combinación lineal: $z = \sum_{i=1}^{n} w_i x_i + b = \mathbf{w}^T \mathbf{x} + b$
- Activación: $a = f(z)$

#### 2.3 Cómputo por capas de la ANN
- Pre-activación: $\mathbf{z}^{(l)} = \mathbf{W}^{(l)} \mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}$
- Activación: $\mathbf{a}^{(l)} = f^{(l)}(\mathbf{z}^{(l)})$
- Donde $\mathbf{W}^{(l)} \in \mathbb{R}^{n_l \times n_{l-1}}$, $\mathbf{b}^{(l)} \in \mathbb{R}^{n_l}$

#### 2.4 Funciones de activación
- ReLU (capas ocultas): $f(z) = \max(0, z)$
- Sigmoid (capa de salida): $\sigma(z) = \frac{1}{1 + e^{-z}}$
- Justificación: ReLU evita el vanishing gradient; sigmoid mapea la salida a probabilidad [0,1]

#### 2.5 Capa de salida y regla de decisión
- Probabilidad predicha: $\hat{p} = \sigma(\mathbf{w}^{(L)T} \mathbf{a}^{(L-1)} + b^{(L)})$
- Clasificación: $\hat{y} = \begin{cases} 1 & \text{si } \hat{p} \geq \tau \\ 0 & \text{si } \hat{p} < \tau \end{cases}$
- Donde $\tau$ es el umbral de decisión (por defecto 0.5, pero ajustable)

#### 2.6 Función de pérdida — Binary Cross-Entropy
$$L(\mathbf{W}, \mathbf{b}) = -\frac{1}{m}\sum_{i=1}^{m}\left[y^{(i)}\ln(\hat{p}^{(i)}) + (1 - y^{(i)})\ln(1 - \hat{p}^{(i)})\right]$$

Justificación: penaliza predicciones confiantes pero incorrectas; derivable; estándar para clasificación binaria con salida sigmoid.

#### 2.7 Optimizador — Adam
- Actualización base: $\theta_{t+1} = \theta_t - \alpha \nabla_\theta L(\theta)$
- Primer momento: $m_t = \beta_1 m_{t-1} + (1-\beta_1)\nabla_\theta L$
- Segundo momento: $v_t = \beta_2 v_{t-1} + (1-\beta_2)(\nabla_\theta L)^2$
- Corrección de sesgo: $\hat{m}_t = \frac{m_t}{1-\beta_1^t}$, $\hat{v}_t = \frac{v_t}{1-\beta_2^t}$
- Actualización: $\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon}\hat{m}_t$

Justificación: combina las ventajas de Momentum y RMSProp; tasa de aprendizaje adaptativa; converge rápido.

#### 2.8 Backpropagation
$$\nabla Error = \left(\frac{\partial L}{\partial f^{(k)}} \frac{\partial f^{(k)}}{\partial z^{(k)}} \frac{\partial z^{(k)}}{\partial W^{(k)}}, \; \frac{\partial L}{\partial f^{(k)}} \frac{\partial f^{(k)}}{\partial z^{(k)}} \frac{\partial z^{(k)}}{\partial b^{(k)}}\right)$$

$$W^{(l)} \leftarrow W^{(l)} - \alpha \frac{\partial L}{\partial W^{(l)}}, \quad b^{(l)} \leftarrow b^{(l)} - \alpha \frac{\partial L}{\partial b^{(l)}}$$

#### 2.9 Estandarización
$$x_{\text{scaled}} = \frac{x - \mu}{\sigma}$$

Donde $\mu$ y $\sigma$ se calculan **únicamente sobre el conjunto de entrenamiento**.

#### 2.10 Métricas de evaluación

**Matriz de confusión:**

|  | Predicho Positivo | Predicho Negativo |
|--|------------------|------------------|
| Real Positivo | TP | FN |
| Real Negativo | FP | TN |

**Métricas derivadas:**
- Accuracy: $\text{Acc} = \frac{TP + TN}{TP + TN + FP + FN}$
- Precision: $\text{Prec} = \frac{TP}{TP + FP}$
- Recall (Sensibilidad): $\text{Recall} = \frac{TP}{TP + FN}$
- F1-Score: $F_1 = 2 \cdot \frac{\text{Prec} \cdot \text{Recall}}{\text{Prec} + \text{Recall}}$

#### 2.11 Regularización — Dropout
$$\tilde{a}^{(l)} = \mathbf{r}^{(l)} \odot \mathbf{a}^{(l)}, \quad r_j \sim \text{Bernoulli}(p)$$

Durante entrenamiento, cada neurona se retiene con probabilidad $p$. Esto previene co-adaptación y reduce overfitting.

---

### Sección 3: Carga de Datos y Análisis Exploratorio (EDA)

**Tipo:** Código + Markdown

#### Imports
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, classification_report,
                             recall_score, accuracy_score, precision_score,
                             f1_score, ConfusionMatrixDisplay)
```

#### Carga de datos
```python
df_train = pd.read_csv('hadron_train.csv', index_col=0)
df_test = pd.read_csv('hadron_test.csv', index_col=0)
```

#### EDA — Análisis y visualizaciones
1. `df_train.info()` — tipos, nulos
2. `df_train.describe()` — estadísticas descriptivas
3. `df_train['class'].value_counts()` — distribución de clases
4. **Histogramas por feature y clase** — subplot 2×3, coloreados por clase, con transparencia
5. **Heatmap de correlación** — `plt.imshow()` o `sns.heatmap()` (matplotlib nativo si no se usa seaborn)
6. **Box plots por feature y clase** — subplot 2×3

#### Análisis markdown
- Identificar features con mayor poder discriminativo (probablemente fAlpha y fConc1 muestran mayor separación)
- Comentar sobre la distribución balanceada de clases
- Identificar posibles outliers

---

### Sección 4: Preprocesamiento de Datos

**Tipo:** Código + Markdown

#### Decisiones y justificaciones

| Decisión | Valor | Justificación |
|----------|-------|---------------|
| Split ratio | 80/20 | El test set ya es el 20% del total original; mantenemos 20% para validación del 80% restante |
| Estratificación | Sí (`stratify=y`) | Preservar la proporción de clases en train y val |
| Random state | 42 | Reproducibilidad |
| Escalado | StandardScaler | Features con rangos muy diferentes; fM3Long tiene negativos; StandardScaler centra en media=0, std=1 |
| Fit del scaler | Solo en train | Evitar data leakage del conjunto de validación/test |
| Semillas | `np.random.seed(42)`, `tf.random.set_seed(42)` | Reproducibilidad de resultados |

#### Código
```python
FEATURES = ['fLength', 'fWidth', 'fConc1', 'fM3Long', 'fAlpha', 'fDist']

X = df_train[FEATURES].values
y = df_train['class'].values

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

X_test = df_test[FEATURES].values
X_test_scaled = scaler.transform(X_test)
```

#### Resultado esperado
- Train: ~8,560 muestras
- Validación: ~2,140 muestras
- Test: 2,676 muestras

---

### Sección 5: Construcción y Entrenamiento de Modelos

**Tipo:** Código + Markdown  
**Estrategia:** Empezar simple, ir aumentando complejidad (tip del instructor).

#### Configuración compartida (fija durante exploración de arquitectura)
- **Optimizer:** Adam (lr=0.001, valores por defecto de β₁=0.9, β₂=0.999)
- **Loss:** binary_crossentropy
- **Epochs:** 200 (con EarlyStopping)
- **Batch size:** 32
- **EarlyStopping:** monitor='val_loss', patience=15, restore_best_weights=True

#### Modelos a entrenar

**Modelo 1 — Baseline mínimo (1 capa oculta, 8 neuronas)**
```
Input(6) → Dense(8, relu) → Dense(1, sigmoid)
```
- Propósito: establecer línea base de rendimiento con la red más simple posible.
- Parámetros: ~73

**Modelo 2 — Capa más ancha (1 capa oculta, 32 neuronas)**
```
Input(6) → Dense(32, relu) → Dense(1, sigmoid)
```
- Propósito: evaluar si más neuronas en una sola capa mejoran el rendimiento.
- Parámetros: ~257

**Modelo 3 — Dos capas ocultas**
```
Input(6) → Dense(32, relu) → Dense(16, relu) → Dense(1, sigmoid)
```
- Propósito: evaluar el impacto de añadir profundidad.
- Parámetros: ~785

**Modelo 4 — Tres capas ocultas con Dropout**
```
Input(6) → Dense(64, relu) → Dropout(0.3) → Dense(32, relu) → Dropout(0.3) → Dense(16, relu) → Dense(1, sigmoid)
```
- Propósito: evaluar si la regularización con dropout mejora generalización.
- Parámetros: ~2,865

**Modelo 5 — Red más profunda y ancha con Dropout**
```
Input(6) → Dense(128, relu) → Dropout(0.3) → Dense(64, relu) → Dropout(0.3) → Dense(32, relu) → Dropout(0.2) → Dense(16, relu) → Dense(1, sigmoid)
```
- Propósito: máxima capacidad de representación con regularización.
- Parámetros: ~12,177

#### Para cada modelo se debe:
1. Construir y compilar con `model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])`
2. Entrenar con `model.fit()` usando EarlyStopping
3. Graficar **curvas de aprendizaje**: 2 paneles (loss train/val, accuracy train/val)
4. Predecir sobre validación: `y_pred = (model.predict(X_val_scaled) >= 0.5).astype(int).flatten()`
5. Calcular métricas: recall, accuracy, precision, F1
6. Mostrar **matriz de confusión** con `ConfusionMatrixDisplay`

---

### Sección 6: Comparación de Modelos

**Tipo:** Código + Markdown

#### Visualizaciones requeridas
1. **Tabla resumen** (DataFrame) con columnas: Modelo, Arquitectura, Params, Accuracy, Recall, Precision, F1, Epochs usados
2. **Curvas de aprendizaje lado a lado** — Figura grande: 5 filas × 2 columnas (loss y accuracy por modelo)
3. **Gráfico de barras** comparando recall entre los 5 modelos

#### Análisis markdown
- ¿Cuáles modelos muestran overfitting? (gap grande entre train y val)
- ¿El dropout reduce efectivamente el overfitting?
- ¿La profundidad adicional mejora o empeora el recall?
- **Seleccionar los top 2-3 candidatos** para la fase de grid search
- Justificar la selección con base en las métricas y curvas observadas

---

### Sección 7: Grid Search de Hiperparámetros

**Tipo:** Código + Markdown

#### Justificación
El grid search se aplica **después** de identificar la mejor arquitectura (tip del instructor). Se usa la arquitectura ganadora de la Sección 6 y se varían los hiperparámetros.

#### Hiperparámetros a buscar

| Hiperparámetro | Valores | Justificación |
|----------------|---------|---------------|
| Learning rate | [0.001, 0.0005, 0.0001] | Rango estándar para Adam; lr más bajo puede mejorar convergencia |
| Batch size | [16, 32, 64] | Batch pequeño → más ruido pero mejor generalización; grande → más estable pero puede converger a mínimos más amplios |
| Dropout rate | [0.2, 0.3, 0.4] | Controla la fuerza de regularización |

**Total: 3 × 3 × 3 = 27 combinaciones**

#### Implementación
Grid search **manual** con loops anidados (no usar scikeras — es librería externa no permitida):

```python
from itertools import product

results_grid = []
for lr, batch_size, dropout_rate in product([0.001, 0.0005, 0.0001], [16, 32, 64], [0.2, 0.3, 0.4]):
    # Fijar semillas para reproducibilidad
    tf.random.set_seed(42)
    np.random.seed(42)
    
    # Construir modelo con la arquitectura ganadora
    model = build_model(dropout_rate, lr)  # función helper
    
    # Entrenar
    history = model.fit(X_train_scaled, y_train, epochs=200, batch_size=batch_size,
                        validation_data=(X_val_scaled, y_val),
                        callbacks=[EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)],
                        verbose=0)
    
    # Evaluar
    y_pred = (model.predict(X_val_scaled, verbose=0) >= 0.5).astype(int).flatten()
    rec = recall_score(y_val, y_pred)
    results_grid.append({'lr': lr, 'batch_size': batch_size, 'dropout': dropout_rate, 'recall': rec, ...})
```

#### Salida
- Tabla con los top 10 combinaciones ordenadas por recall
- Heatmap de resultados (por ejemplo, lr vs batch_size, coloreado por recall promedio)
- Identificar la **mejor combinación de hiperparámetros**

---

### Sección 8: Optimización del Umbral de Decisión

**Tipo:** Código + Markdown  
**Esta sección es CLAVE para maximizar recall.**

#### Justificación
Por defecto, la clasificación binaria usa umbral $\tau = 0.5$. Sin embargo, dado que el 50% de la nota depende del recall, conviene explorar umbrales más bajos que capturen más hadrones (menos FN) a costa de más FP.

- **Bajar el umbral** → más predicciones positivas (hadrón) → mayor recall, menor precision
- El tradeoff es aceptable porque la nota **solo mide recall**, no precision ni accuracy

#### Implementación
```python
y_val_prob = best_model.predict(X_val_scaled, verbose=0).flatten()

thresholds = np.arange(0.30, 0.55, 0.01)
threshold_results = []
for t in thresholds:
    y_pred_t = (y_val_prob >= t).astype(int)
    threshold_results.append({
        'threshold': t,
        'recall': recall_score(y_val, y_pred_t),
        'precision': precision_score(y_val, y_pred_t),
        'accuracy': accuracy_score(y_val, y_pred_t)
    })
```

#### Visualización
- Gráfica: recall, precision y accuracy vs umbral (3 líneas en un mismo plot)
- Marcar el umbral seleccionado con una línea vertical

#### Criterio de selección
Elegir el umbral que **maximice recall** mientras mantenga accuracy por encima de ~0.70. Justificar la decisión en el contexto de la rúbrica.

---

### Sección 9: Identificación del Modelo Final

**Tipo:** Markdown + Código

#### Contenido markdown obligatorio
Declarar de forma **clara y explícita**:
- Arquitectura final (capas, neuronas, activaciones, dropout)
- Optimizer y learning rate
- Loss function
- Batch size
- Epochs utilizados (con early stopping)
- Umbral de decisión elegido
- Método de estandarización

#### Contenido código
- `best_model.summary()` — tabla de la arquitectura
- Matriz de confusión final sobre validación
- `classification_report(y_val, y_pred_final)` — reporte completo
- Valor final de recall en validación
- Verificar que recall > 0.7

---

### Sección 10: Análisis Contextual del Modelo Final

**Tipo:** Markdown  
**Vale 10% de la nota total (20% del informe). Debe ser profundo.**

#### Contenido obligatorio

**Interpretación clínica de la matriz de confusión:**

| Resultado | Significado Clínico | Consecuencia |
|-----------|---------------------|-------------|
| **TP** (hadrón → hadrón) | Haz terapéutico correctamente identificado | Llega al gantry, paciente recibe tratamiento adecuado |
| **TN** (gamma → gamma) | Radiación gamma correctamente filtrada | Se descarta correctamente, paciente protegido |
| **FP** (gamma → hadrón) | Gamma pasa como hadrón | **PELIGROSO**: paciente recibe radiación gamma sin beneficio del pico de Bragg, posible daño a tejido sano |
| **FN** (hadrón → gamma) | Hadrón descartado como gamma | Pérdida de un haz terapéutico válido — reduce eficiencia del tratamiento pero no daña al paciente |

**Discusión del trade-off recall/precision:**
- Un modelo con alto recall para hadrones minimiza FN (pocos hadrones desperdiciados)
- Pero puede aumentar FP (más gammas pasan como hadrones)
- En un sistema real, se podría añadir una segunda etapa de verificación para los FP
- Los FN son pérdidas irrecuperables de haces terapéuticos

**Limitaciones del modelo:**
- Solo 6 features del detector — podrían no capturar toda la física del fenómeno
- El modelo asume que la distribución de features es estacionaria
- No se modela la incertidumbre en las predicciones

**Features más discriminativas:**
- Análisis basado en los pesos de la primera capa o en observaciones del EDA

---

### Sección 11: Generación y Validación de predictions.csv

**Tipo:** Código

#### Generación
```python
y_test_prob = best_model.predict(X_test_scaled, verbose=0).flatten()
y_test_pred = (y_test_prob >= THRESHOLD).astype(int)

df_predictions = df_test[FEATURES].copy()
df_predictions['prediction'] = y_test_pred
df_predictions.to_csv('predictions.csv', index=False)
```

#### Validación con check_requirements()
```python
import importlib.util
spec = importlib.util.spec_from_file_location("helpers", "helpers - Copy.py")
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
helpers.check_requirements()
```

#### Sanity checks
- Verificar que predictions.csv tiene exactamente 2676 filas
- Verificar que la columna "prediction" contiene solo 0s y 1s
- Comparar distribución de predicciones vs distribución de entrenamiento (~50/50)
- Verificar que `check_requirements()` pasa sin errores

---

## 6. Estrategia de Maximización de Recall (Priorizada)

| Prioridad | Técnica | Impacto Esperado | Descripción |
|-----------|---------|-----------------|-------------|
| 1 | **Ajuste de umbral** | ALTO | Bajar umbral de 0.5 a ~0.40-0.45; captura más hadrones a costa de precision |
| 2 | **Arquitectura óptima** | MEDIO-ALTO | Red con suficiente capacidad + dropout para generalizar bien |
| 3 | **Grid search** | MEDIO | Encontrar la combinación lr/batch/dropout que maximice recall |
| 4 | **Class weights** | BAJO (respaldo) | `class_weight={0: 1.0, 1: 1.2}` si recall sigue bajo; sesgar el entrenamiento hacia hadrones |

---

## 7. Orden de Implementación

| Fase | Secciones | Tipo de trabajo | Dependencias |
|------|-----------|----------------|-------------|
| 1 | 1, 2 | Markdown/LaTeX (informe) | Ninguna |
| 2 | 3 | Código (imports, carga, EDA) | Ninguna |
| 3 | 4 | Código (preprocesamiento) | Fase 2 |
| 4 | 5 | Código (5 modelos) | Fase 3 |
| 5 | 6 | Código + Markdown (comparación) | Fase 4 |
| 6 | 7 | Código (grid search) | Fase 5 |
| 7 | 8 | Código (threshold tuning) | Fase 6 |
| 8 | 9 | Markdown + Código (modelo final) | Fase 7 |
| 9 | 10 | Markdown (análisis contextual) | Fase 8 |
| 10 | 11 | Código (predictions.csv) | Fase 8 |

---

## 8. Checklist de Verificación Final

- [ ] Notebook ejecuta de inicio a fin sin errores
- [ ] Todas las ecuaciones están en formato LaTeX (`$$...$$`)
- [ ] Las 6 secciones requeridas del informe están presentes
- [ ] Se muestran múltiples modelos con tablas y gráficas comparativas
- [ ] Curvas de aprendizaje incluidas para cada modelo
- [ ] Matriz de confusión incluida
- [ ] Recall reportado explícitamente
- [ ] Modelo final con arquitectura e hiperparámetros claramente identificados
- [ ] Análisis contextual desde hadronterapia con profundidad
- [ ] `predictions.csv` tiene 7 columnas, 2676 filas, sin índice
- [ ] `check_requirements()` pasa sin errores
- [ ] Recall en validación > 0.7
- [ ] Solo se usan librerías permitidas (TensorFlow/Keras, NumPy, Pandas, Matplotlib, Scikit-Learn, Scipy)
- [ ] Código limpio con buenas prácticas de programación
