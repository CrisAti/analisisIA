# 📘 Propuestas de Métrica de Transferencia para Clustering

## Título de la tesis

**Diseño de una Métrica de Transferencia para la Utilización de Conjuntos de Datos de Clasificación en Tareas de Clustering**

---

# 🧠 Contexto General

Esta investigación se fundamenta en tres pilares:

* **CLM (Cluster-Label Matching)** → mide qué tan bien las etiquetas representan la estructura natural
* **JMDS (Joint Model-Data Structure)** → mide la confiabilidad de asignaciones individuales
* **Clustering con restricciones de tamaño** → introduce condiciones realistas (balance, capacidad)

## Problema identificado

Las métricas existentes:

* No consideran restricciones estructurales
* No combinan análisis global (dataset) con análisis local (instancias)

---

# 🚀 Propuesta 1: TCMS

## Transferability under Constrained Metric Structure

### 🎯 Objetivo

Evaluar qué tan transferible es un dataset de clasificación a clustering bajo restricciones de tamaño.

### 📐 Fórmula

```
TCMS = α · CLM + β · StructuralConfidence − γ · SizeViolation
```

---

## 🔹 Componentes

### 1. CLM (base)

* Basado en CHA
* Evalúa alineación entre clases y clusters

---

### 2. Structural Confidence (inspirado en JMDS)

```
SC(x_i) = Separation(x_i) · Stability(x_i)
```

#### Separation

Distancia entre el cluster asignado y el más cercano:

```
Separation = d(x_i, C_j) − d(x_i, C_k)
```

#### Stability

Consistencia del clustering:

```
Stability = frecuencia de asignación al mismo cluster
```

---

### 3. Size Violation

```
SizeViolation = Σ max(0, |C_k| − U_k) + max(0, L_k − |C_k|)
```

---

## 💡 Interpretación

* Alto TCMS → dataset adecuado para clustering real
* Bajo TCMS → dataset problemático bajo restricciones

---

# 🧠 Propuesta 2: C-CLM

## Constrained CLM

### 🎯 Objetivo

Modificar CLM incorporando balance de clusters

### 📐 Fórmula

```
C-CLM = CHA · exp(−λ · SizeVariance)
```

### 📊 SizeVariance

```
Var(|C_1|, ..., |C_K|)
```

---

## 💡 Interpretación

* Penaliza datasets desbalanceados
* Fácil implementación

---

# 🚀 Propuesta 3: JMDS Adaptado a Clustering

### 🎯 Objetivo

Evaluar la clusterabilidad a nivel de muestra

### 📐 Fórmula

```
JMDS_c(x_i) = LPG_c(x_i) · Stability_c(x_i)
```

---

## 🔹 Componentes

### LPG_c

```
log P(cluster_i) − log P(cluster_j)
```

---

### Stability_c

Consistencia en múltiples ejecuciones

---

## 📊 Score global

```
DatasetScore = (1/N) Σ JMDS_c(x_i)
```

---

# 🧠 Propuesta 4: TDS

## Transfer Difficulty Score

### 🎯 Objetivo

Medir dificultad de transición clasificación → clustering

### 📐 Fórmula

```
TDS = H(Labels | Clusters) + SizePenalty
```

---

## 🔹 Componentes

### Entropía condicional

Mide mezcla de clases dentro de clusters

---

### SizePenalty

Penalización por restricciones

---

## 💡 Interpretación

* Alto TDS → difícil clusterizar
* Bajo TDS → buena estructura

---

# 🚀 Propuesta 5: HCM

## Hybrid CLM-JMDS Metric

### 🎯 Objetivo

Combinar evaluación global y local

### 📐 Fórmula

```
HCM = CHA · (1/N Σ JMDS(x_i))
```

---

## 💡 Interpretación

* CHA → estructura global
* JMDS → confianza local

---

# 🧪 Evaluación Experimental

## Dataset sugeridos

* MNIST
* CIFAR-10
* Office-31

---

## Experimentos

1. Clustering normal vs restringido
2. Comparación con métricas tradicionales
3. Correlación con accuracy

---

# 🧠 Conclusión

Estas propuestas permiten:

* Extender CLM hacia escenarios reales
* Integrar confiabilidad tipo JMDS
* Introducir restricciones estructurales

---

# 🚀 Contribución principal

"Diseño de una métrica híbrida que integra alineación estructural, confiabilidad local y restricciones de tamaño para evaluar la transferibilidad de datasets de clasificación a clustering."
