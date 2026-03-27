# Propuestas de Métricas de Transferencia para Clustering con Restricciones de Tamaño
### Tesis: "Diseño de una Métrica de Transferencia para la Utilización de Conjuntos de Datos de Clasificación en Tareas de Clustering"

---

## Introducción

Se presentan tres propuestas metodológicas para diseñar una métrica de transferencia que determine qué tan apto es un dataset etiquetado (fuente) para ser utilizado en tareas de clustering con restricción de tamaño (size-constrained clustering). Cada propuesta integra, en distinto grado, los conceptos del estado del arte de **CLM** (Cluster-Label Matching, Jeon et al., 2025) y **JMDS** (Joint Model-Data Structure score).

---

## Propuesta 1 — Métrica de Transferibilidad Estructural ($T\text{-}CH_A$)

### Concepto Central

Esta propuesta adapta el Índice de Calinski-Harabasz Ajustado ($CH_A$) del framework CLM para que no solo mida si las etiquetas coinciden con los clusters naturales, sino si esa estructura **sobrevive** a una restricción de tamaño impuesta externamente.

La idea clave es medir la **"pérdida de cohesión"** o **"pérdida de energía estructural"** que ocurre cuando se obliga al dataset a agruparse en clusters de tamaño específico (por ejemplo, clusters perfectamente balanceados). Si la estructura original del dataset ya es cercana a la estructura restringida, la pérdida será pequeña y el dataset será altamente transferible.

### Funcionamiento

El cálculo se realiza en tres pasos:

**Paso 1 — Calcular el $CH_A$ sin restricciones:**

Se aplica el $CH_A$ estándar del framework CLM sobre el dataset fuente $D_s$ con $K$ clusters:

$$CH_A^{\text{libre}}(D_s, K) = \sigma\!\left(\beta_0 + \beta_1 \cdot \log\!\left(\frac{B_K^{\text{libre}} / (K-1)}{W_K^{\text{libre}} / (n-K)}\right)\right) \in [0, 1]$$

donde $B_K^{\text{libre}}$ y $W_K^{\text{libre}}$ son las varianzas inter e intra-cluster del agrupamiento libre (sin restricción).

**Paso 2 — Calcular el $CH$ bajo restricción de tamaño:**

Se ejecuta un algoritmo de clustering con restricción de tamaño (e.g., Balanced K-Means, Constrained K-Means) que fuerza $|C_k| \in [n_k^{\min}, n_k^{\max}]$, obteniendo la partición restringida $\hat{\mathcal{C}}^{\text{rest}}$. Con esta partición se calcula el índice Calinski-Harabasz crudo:

$$CH^{\text{rest}}(D_s, K) = \frac{B_K^{\text{rest}} / (K-1)}{W_K^{\text{rest}} / (n-K)}$$

**Paso 3 — Calcular la métrica de transferencia:**

$$\boxed{T\text{-}CH_A(D_s, K) = \frac{CH^{\text{rest}}(D_s, K)}{CH^{\text{libre}}(D_s, K)}}$$

Dado que $CH^{\text{rest}} \leq CH^{\text{libre}}$ siempre (agregar restricciones no puede mejorar la cohesión óptima), la métrica satisface $T\text{-}CH_A \in (0, 1]$.

**Interpretación del valor:**
- $T\text{-}CH_A \approx 1$: La estructura natural del dataset ya es compatible con las restricciones de tamaño. Alto potencial de transferencia.
- $T\text{-}CH_A \approx 0$: Imponer la restricción de tamaño destruye casi toda la cohesión del clustering. Bajo potencial de transferencia.

### Propiedades y Discusión

| Propiedad | Valor |
|---|---|
| Rango | $(0, 1]$ |
| Requiere ejecutar clustering restringido | Sí |
| Parámetros libres | 0 (usa calibración previa de CLM) |
| Interpretabilidad | Alta (razón de cohesión) |
| Costo computacional | Medio (un paso de clustering restringido) |

**Fortaleza principal:** Directamente interpretable como la fracción de cohesión que se preserva bajo la restricción. No requiere supuestos paramétricos adicionales.

**Limitación principal:** Depende de la calidad del algoritmo de clustering restringido usado en el Paso 2 — diferentes algoritmos pueden producir distintos $CH^{\text{rest}}$. Se recomienda fijar un algoritmo estándar (e.g., Balanced K-Means con semilla aleatoria fija) para garantizar reproducibilidad.

**Relación con CLM:** Los axiomas A1–A4 se heredan parcialmente a través de $CH_A^{\text{libre}}$. Sin embargo, $CH^{\text{rest}}$ en el numerador no pasa por el ajuste logístico de CLM, por lo que la ratio puede no satisfacer A4 (invariancia de rango) de forma estricta. Una variante axiomáticamente más rigurosa es aplicar el ajuste logístico también a $CH^{\text{rest}}$.

---

## Propuesta 2 — Score de Confianza Transferida ($JMDS\text{-}T$)

### Concepto Central

Utilizando la arquitectura del JMDS (Joint Model-Data Structure score), se diseña una métrica que evalúa la confianza de que una etiqueta de clasificación sea un "buen cluster" bajo restricciones de tamaño. En este contexto:

- El **"Modelo"** es el dataset de clasificación original: las etiquetas $y_i$ actúan como pseudo-modelo de referencia.
- La **"Estructura"** es el resultado del clustering restringido: la partición $\hat{\mathcal{C}}^{\text{rest}}$.

La métrica evalúa, por cada muestra, si su etiqueta original coincide con su asignación en el clustering restringido, ponderando por la certeza de dicha asignación.

### Componentes del $JMDS\text{-}T$

#### Componente 1 — $LPG^{\text{rest}}$: Log-Probability Gap Restringido

En el JMDS original, el $LPG$ usa un GMM estándar para medir la separación entre la componente primaria y secundaria de cada muestra. En $JMDS\text{-}T$, se modifica para incorporar el costo de la restricción de tamaño.

Se ajusta un GMM con $K$ componentes con penalización de tamaño. Esto puede implementarse de dos formas:

**Opción A (Hard):** Usar directamente las probabilidades de pertenencia del algoritmo de clustering restringido $\hat{r}_{ik} = p(z_i = k \mid x_i, \hat{\mathcal{C}}^{\text{rest}})$.

**Opción B (Soft):** Ajustar un GMM estándar y penalizar el log-likelihood de cada componente por el costo de violar la restricción:

$$\tilde{p}(x_i \mid \text{comp}_k) = p(x_i \mid \text{comp}_k) \cdot \exp\!\left(-\mu \cdot \text{cost}(k, \mathcal{C}^{\text{rest}})\right)$$

donde $\text{cost}(k, \mathcal{C}^{\text{rest}}) = \max\!\left(0,\, \frac{|C_k|}{n} - u_k\right) + \max\!\left(0,\, l_k - \frac{|C_k|}{n}\right)$ es la violación de la restricción del cluster $k$, y $\mu > 0$ controla la fuerza de la penalización.

El $LPG^{\text{rest}}$ de cada muestra es:

$$LPG^{\text{rest}}(x_i) = \log \tilde{p}(x_i \mid \text{comp}_{z^*_i}) - \log \tilde{p}(x_i \mid \text{comp}_{z^{**}_i})$$

Un $LPG^{\text{rest}}(x_i)$ alto indica que $x_i$ pertenece con alta certeza al cluster restringido $z^*_i$, incluso considerando el costo de la restricción.

#### Componente 2 — $MPPL^{\text{rest}}$: Probabilidad del Modelo sobre la Etiqueta Restringida

Evalúa qué tan probable es que la **etiqueta de clasificación original** $y_i$ coincida con el **cluster restringido asignado** $\hat{z}_i^{\text{rest}}$:

$$MPPL^{\text{rest}}(x_i) = p(y_i = \hat{z}_i^{\text{rest}})$$

Dado que las etiquetas son discretas, esta probabilidad se estima como la proporción de muestras de la clase $y_i$ que caen en el cluster $\hat{z}_i^{\text{rest}}$ (pureza local):

$$MPPL^{\text{rest}}(x_i) = \frac{|\{x_j : y_j = y_i,\; \hat{z}_j^{\text{rest}} = \hat{z}_i^{\text{rest}}\}|}{|\{x_j : \hat{z}_j^{\text{rest}} = \hat{z}_i^{\text{rest}}\}|}$$

Valores cercanos a 1 indican que el cluster restringido al que fue asignada $x_i$ está dominado por su clase, i.e., la etiqueta original es una guía confiable para ese cluster.

#### Métrica por Muestra y Global

El score por muestra, siguiendo la arquitectura JMDS, es el producto:

$$JMDS\text{-}T(x_i) = \sigma\!\left(LPG^{\text{rest}}(x_i)\right) \cdot MPPL^{\text{rest}}(x_i) \in [0, 1]$$

donde $\sigma(\cdot)$ es la función sigmoidea que normaliza el $LPG$ en $[0,1]$.

La **métrica global de transferencia** del dataset es:

$$\boxed{JMDS\text{-}T(D_s, K) = \frac{1}{n}\sum_{i=1}^{n} JMDS\text{-}T(x_i) \in [0, 1]}$$

### Interpretación

- $JMDS\text{-}T \approx 1$: Las etiquetas de clasificación son guías confiables para el clustering restringido. Las muestras son asignadas con alta certeza a clusters que coinciden con sus clases originales, incluso bajo la restricción.
- $JMDS\text{-}T \approx 0$: El clustering restringido rompe la correspondencia entre etiquetas y clusters. El dataset no es transferible en este escenario.

**Uso adicional — ponderación de muestras:** Un subproducto valioso de $JMDS\text{-}T$ es el vector de scores por muestra $\{JMDS\text{-}T(x_i)\}_{i=1}^n$, que puede usarse para **ponderar las muestras** durante el entrenamiento de un clasificador auxiliar o para identificar qué muestras son "anclas confiables" en el espacio de clustering restringido.

### Propiedades y Discusión

| Propiedad | Valor |
|---|---|
| Rango | $[0, 1]$ |
| Requiere ejecutar clustering restringido | Sí |
| Parámetros libres | $\mu$ (fuerza de penalización) |
| Interpretabilidad | Alta (confianza promedio por muestra) |
| Subproducto útil | Pesos por muestra para uso downstream |
| Costo computacional | Medio-Alto (GMM + clustering restringido) |

**Fortaleza principal:** Es la única propuesta que produce una puntuación **por muestra**, lo que permite identificar qué regiones del espacio de features son transferibles y cuáles no. Esto tiene aplicaciones directas en selección de instancias y curriculum learning.

**Limitación principal:** El parámetro $\mu$ requiere calibración. Además, la estimación de $MPPL^{\text{rest}}$ por pureza local puede ser ruidosa cuando los clusters son pequeños.

---

## Propuesta 3 — Extensión Axiomática: Quinto Axioma de Invariancia al Balance de Tamaño

### Concepto Central

En lugar de diseñar una métrica nueva desde cero, esta propuesta **extiende formalmente el framework axiomático de CLM** con un quinto axioma que hace explícita la necesidad de penalizar datasets cuya distribución de etiquetas es incompatible con las restricciones de tamaño objetivo.

### El Quinto Axioma

> **A5 — Invariancia al Balance de Tamaño (Size-Balance Invariance):**
> Sea $D$ un dataset con distribución de tamaños de clase $\boldsymbol{\pi}^D = (\pi_1^D, \ldots, \pi_C^D)$ y $D'$ un dataset con $\boldsymbol{\pi}^{D'}$. Sea $\mathbf{u}_K = (1/K, \ldots, 1/K)$ la distribución objetivo (clusters uniformes). Si $\boldsymbol{\pi}^D$ y $\boldsymbol{\pi}^{D'}$ difieren únicamente en su distancia a $\mathbf{u}_K$, entonces la métrica de transferencia $\mathcal{M}$ debe satisfacer:
>
> $$\Delta(\boldsymbol{\pi}^D, \mathbf{u}_K) < \Delta(\boldsymbol{\pi}^{D'}, \mathbf{u}_K) \implies \mathcal{M}(D, K) > \mathcal{M}(D', K)$$
>
> donde $\Delta$ es una medida de discrepancia (e.g., distancia $W_1$, JSD, variación total).

En palabras: **si el dataset $D$ tiene una distribución de etiquetas más cercana a la distribución objetivo de los clusters, entonces debe recibir una mayor puntuación de transferencia**, todo lo demás igual.

### Motivación del Axioma

Los axiomas A1–A4 de CLM garantizan que una métrica sea válida para comparar datasets en términos de separabilidad y estructura geométrica. Sin embargo, ninguno de ellos contempla la **compatibilidad distribucional** entre las etiquetas fuente y las restricciones del clustering objetivo.

Un ejemplo que motiva A5: consideremos dos datasets con la misma separabilidad ($CH_A$ idéntico):
- $D$: clases de tamaños $(50\%, 50\%)$, objetivo $K=2$ clusters iguales → distribución ya compatible.
- $D'$: clases de tamaños $(90\%, 10\%)$, objetivo $K=2$ clusters iguales → distribución incompatible.

Sin A5, cualquier métrica basada solo en A1–A4 asignaría igual puntuación a $D$ y $D'$. Con A5, $\mathcal{M}(D, K) > \mathcal{M}(D', K)$ necesariamente.

### Implementación — Factor de Normalización por Entropía

A5 se implementa modificando los **protocolos de ajuste T1–T4 de Jeon et al.** para incluir un factor de normalización basado en la compatibilidad distribucional. Se proponen dos implementaciones:

#### Implementación A — Factor Multiplicativo Entrópico

Definir el factor de balance como:

$$\Phi(D_s, K) = \exp\!\left(-\lambda \cdot \text{JSD}(\boldsymbol{\pi}^{\text{clase}}\,\|\,\mathbf{u}_K)\right) \in (0, 1]$$

donde $\boldsymbol{\pi}^{\text{clase}} = (|C_1|/n, \ldots, |C_C|/n)$ es la distribución empírica de tamaños de clase y $\mathbf{u}_K = (1/K, \ldots, 1/K)$ es la distribución objetivo. El parámetro $\lambda > 0$ controla la sensibilidad.

La métrica A5-compatible es:

$$\mathcal{M}_{A5}(D_s, K) = CH_A(D_s, K) \cdot \Phi(D_s, K)$$

#### Implementación B — Ajuste Logístico con Covariable de Balance (preferida)

Incorporar A5 directamente dentro del ajuste logístico de CLM añadiendo el desequilibrio como covariable:

$$\mathcal{M}_{A5}(D_s, K) = \sigma\!\left(\beta_0 + \beta_1 \cdot \log(CH_K) + \beta_2 \cdot \log\!\left(1 - W_1(\boldsymbol{\pi}^{\text{clase}}, \mathbf{u}_K)\right)\right)$$

donde $W_1(\boldsymbol{\pi}^{\text{clase}}, \mathbf{u}_K) = \frac{1}{2}\sum_{k}|\pi_k^{\text{clase}} - 1/K|$ es la distancia de Wasserstein discreta entre la distribución de clases y la distribución objetivo.

**Ventaja de la Implementación B:** La penalización de balance queda integrada en la misma función logística que calibra CLM, por lo que hereda automáticamente A1–A4. Los parámetros $(\beta_0, \beta_1, \beta_2)$ se calibran con el mismo protocolo de Jeon et al. ampliado.

### Relación con los Protocolos T1–T4 de CLM

Los protocolos de ajuste de CLM (T1: escala, T2: rotación, T3: cardinalidad, T4: etiquetas) se extienden con:

> **T5 — Protocolo de Balance de Tamaño:** Para calibrar $\beta_2$, se generan pares de datasets con la misma separabilidad pero diferente distribución de tamaños de clase. Se verifica que $\mathcal{M}_{A5}$ asigna mayor puntaje al dataset más balanceado, y se ajusta $\beta_2$ hasta que la tasa de acierto sobre el conjunto de calibración supere el 95%.

### Propiedades y Discusión

| Propiedad | Valor |
|---|---|
| Rango | $[0, 1]$ |
| Axiomas satisfechos | A1–A5 |
| Parámetros libres | 3 ($\beta_0, \beta_1, \beta_2$) |
| Requiere ejecutar clustering restringido | No |
| Interpretabilidad | Alta (extensión natural de CLM) |
| Rigor teórico | Alto (contribución axiomática formal) |

**Fortaleza principal:** Es la única propuesta que extiende formalmente un framework teórico existente. La demostración de que A5 es **independiente** de A1–A4 (i.e., ninguno de ellos implica A5) es en sí misma una contribución teórica de la tesis.

**Limitación principal:** Al no ejecutar clustering restringido, evalúa la compatibilidad distribucional a partir de las etiquetas originales, que pueden no reflejar la estructura geométrica real. Es posible que un dataset con etiquetas desbalanceadas tenga clusters geométricos naturalmente balanceados.

**Recomendación:** Combinar la Propuesta 3 con la Propuesta 1 o 2 para capturar tanto la compatibilidad distribucional (A5) como el efecto geométrico real de la restricción.

---

## Cuadro Comparativo de las Tres Propuestas

| Característica | Propuesta 1 ($T\text{-}CH_A$) | Propuesta 2 ($JMDS\text{-}T$) | Propuesta 3 (Axioma A5) |
|---|---|---|---|
| **Base teórica** | CLM ($CH_A$) | JMDS (LPG + MPPL) | CLM extendido |
| **Rango** | $(0, 1]$ | $[0, 1]$ | $[0, 1]$ |
| **Requiere clustering restringido** | Sí | Sí | No |
| **Axiomas CLM** | A1–A4 (parcial) | Ninguno | A1–A5 |
| **Score por muestra** | No | Sí | No |
| **Parámetros libres** | 0 | 1 ($\mu$) | 3 ($\beta_0,\beta_1,\beta_2$) |
| **Costo computacional** | Medio | Medio-Alto | Bajo-Medio |
| **Interpretabilidad** | Alta | Alta | Alta |
| **Rigor teórico** | Medio | Medio | Alto |
| **Originalidad** | Media | Media-Alta | Alta |
| **Uso recomendado** | Baseline interpretable | Ponderación de muestras | Contribución teórica principal |

---

## Cuadro Comparativo: Enfoques CLM vs. JMDS para la Métrica de Transferencia

| Característica | Enfoque Basado en CLM | Enfoque Basado en JMDS |
|---|---|---|
| **Fortaleza principal** | Validez estadística y teórica sólida (axiomas A1–A4) | Muy efectivo para manejar ruido y etiquetas "sucias" |
| **Cálculo** | Promedios de distancias euclidianas al cuadrado ($d^2$) | Log-likelihood de GMM y probabilidades de modelo |
| **Uso en clustering** | Ideal para comparar qué tan "bueno" es un dataset frente a otro | Ideal para dar pesos a las muestras durante el entrenamiento |
| **Salida** | Escalar global del dataset | Score por muestra + escalar global |
| **Necesita modelo fuente** | No | Opcional (MPPL requiere clasificador) |
| **Sensibilidad a dimensionalidad** | Controlada por axioma A2 (shift invariance) | Controlada por el GMM (puede fallar en muy alta dimensión) |
| **Calibración** | Protocolo T1–T4 con datos humanos | Depende del parámetro $\mu$ y del umbral $\tau$ |

---

## Estrategia de Integración Recomendada para la Tesis

Las tres propuestas no son excluyentes. Se recomienda la siguiente arquitectura integrada:

```
Dataset fuente D_s
        │
        ├─── Propuesta 3 (A5): Compatibilidad distribucional
        │    → Penalización rápida antes de ejecutar clustering
        │
        ├─── Propuesta 1 (T-CH_A): Cohesión geométrica bajo restricción
        │    → Medición de la "pérdida de energía" estructural
        │
        └─── Propuesta 2 (JMDS-T): Confianza por muestra
             → Identificación de muestras ancla y muestras problemáticas
                      │
                      ▼
            ITCR(D_s, K) = f(T-CH_A, JMDS-T, Φ_{A5})
```

La métrica final $\text{ITCR}$ puede ser una fusión ponderada (Camino 2 del documento metodológico) o una extensión axiomática (Camino 3), donde cada propuesta contribuye uno de los tres factores del producto.

---

*Documento de propuestas metodológicas para discusión con el director de tesis.*