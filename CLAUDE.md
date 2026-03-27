# Índice de Transferibilidad para Clustering con Restricciones (ITCR)
### Propuesta Metodológica para Tesis de Maestría/Doctorado
**"Diseño de una Métrica de Transferencia para la Utilización de Conjuntos de Datos de Clasificación en Tareas de Clustering"**

---

## 1. Definición del Problema y Notación

Sea $D_s = \{(x_i, y_i)\}_{i=1}^{n}$ un dataset fuente etiquetado con $n$ muestras en $\mathbb{R}^d$, $C$ clases y etiquetas $y_i \in \{1, \ldots, C\}$. El objetivo es cuantificar qué tan apto es $D_s$ para ser utilizado en una tarea de clustering con $K$ grupos bajo restricciones de tamaño.

**Notación clave:**

| Símbolo | Descripción |
|---|---|
| $n, d, C$ | Muestras, dimensiones, clases del dataset fuente |
| $K$ | Número de clusters objetivo |
| $\boldsymbol{\pi} = (\pi_1, \ldots, \pi_K)$ | Pesos del GMM ajustado con $K$ componentes |
| $\mathbf{u} = (1/K, \ldots, 1/K)$ | Distribución objetivo (clusters balanceados) |
| $\Pi$ | Caja factible de distribuciones bajo restricciones $[n_k^{\min}, n_k^{\max}]$ |
| $CH_A$ | Índice Calinski-Harabasz Ajustado (CLM) |
| $SBC$ | Size Balance Compatibility (penalización de tamaño) |
| $SCW$ | Sample Confidence Weight (confianza muestral, inspirado en JMDS) |
| $BMF$ | Boundary Mass Fraction (fracción de muestras en frontera) |

La métrica propuesta toma la forma general:

$$\text{ITCR}(D_s, K) \in [0, 1]$$

donde valores cercanos a 1 indican alta transferibilidad y cercanos a 0 indican baja aptitud para el escenario de clustering restringido.

---

## 2. Los Tres Componentes Base

Antes de describir los caminos de fusión, se definen los tres bloques funcionales que los comparten.

### 2.1 Separabilidad: $CH_A$ (heredado de CLM)

Se aplica el framework CLM pero ajustando el GMM con $K$ componentes (el número de clusters objetivo, no las $C$ clases originales), porque interesa la estructura natural en $K$ grupos:

$$CH_A(D_s, K) = \sigma\!\left(\beta_0 + \beta_1 \cdot \log\!\left(\frac{B_K / (K-1)}{W_K / (n-K)}\right)\right) \in [0,1]$$

donde:
- $B_K$ es la varianza inter-cluster (traza de la matriz de dispersión entre centroides del GMM)
- $W_K$ es la varianza intra-cluster (suma de dispersiones dentro de cada componente)
- $\sigma(\cdot)$ es la función logística calibrada con percepción humana (axioma A4 de CLM)
- $\beta_0, \beta_1$ son parámetros calibrados sobre datasets benchmark

**Axiomas CLM heredados:**
- **A1 (Invariancia a cardinalidad):** La ratio $B_K/(K-1)$ sobre $W_K/(n-K)$ es invariante en escala de $n$.
- **A2 (Shift Invariance):** El protocolo de distancias exponenciales del GMM mitiga la concentración en alta dimensionalidad.
- **A3 (Invariancia a cardinalidad de clases):** El promedio par-a-par de los $K$ grupos evita sesgo por número de clusters.
- **A4 (Invariancia de rango):** La función logística normaliza estrictamente en $[0, 1]$.

### 2.2 Balance de Tamaño: $SBC$

Este es el aporte central de la tesis. Mide cuán cerca están los pesos naturales del GMM de la distribución objetivo.

#### Opción A — Distancia de Wasserstein Discreta (recomendada)

$$SBC_W(D_s, K) = 1 - \frac{W_1(\boldsymbol{\pi},\, \mathbf{u})}{W_1^{\max}}$$

donde la distancia $W_1$ discreta entre $\boldsymbol{\pi}$ y $\mathbf{u} = (1/K, \ldots, 1/K)$ es:

$$W_1(\boldsymbol{\pi}, \mathbf{u}) = \frac{1}{2}\sum_{k=1}^{K}\left|\pi_{(k)} - \frac{1}{K}\right|$$

y la normalización exacta es:

$$W_1^{\max} = 1 - \frac{1}{K}$$

de modo que $SBC_W = 1$ cuando $\boldsymbol{\pi} = \mathbf{u}$ (balance perfecto) y $SBC_W = 0$ cuando toda la masa recae en un único cluster.

**Ventaja:** Interpretación directa — penaliza la masa total que "sobraría o faltaría" para equilibrar los clusters. Sensible a la magnitud del desvío.

#### Opción B — Divergencia Jensen-Shannon

$$SBC_{JS}(D_s, K) = 1 - \frac{\text{JSD}(\boldsymbol{\pi}\,\|\,\mathbf{u})}{\log 2}$$

donde $\text{JSD}(\boldsymbol{\pi} \| \mathbf{u}) = \frac{1}{2}KL(\boldsymbol{\pi} \| \mathbf{m}) + \frac{1}{2}KL(\mathbf{u} \| \mathbf{m})$, con $\mathbf{m} = \frac{\boldsymbol{\pi} + \mathbf{u}}{2}$.

**Ventaja:** Más suave que $W_1$ para desviaciones pequeñas; simétrica y acotada. **Desventaja:** Menos interpretable que la distancia de transporte.

#### Generalización a Restricciones de Intervalo

Para restricciones de la forma $n_k \in [n_k^{\min}, n_k^{\max}]$, el espacio factible es una caja dentro del símplex:

$$\Pi = \left\{\boldsymbol{\pi} \in \Delta_K : l_k \leq \pi_k \leq u_k,\; l_k = \frac{n_k^{\min}}{n},\; u_k = \frac{n_k^{\max}}{n}\right\}$$

La penalización generalizada es la distancia del vector GMM a la caja factible:

$$SBC_{\text{proj}}(D_s, K) = 1 - \|\boldsymbol{\pi} - \text{proj}_{\Pi}(\boldsymbol{\pi})\|_1$$

donde $\text{proj}_{\Pi}(\boldsymbol{\pi}) = \arg\min_{\mathbf{q} \in \Pi} \|\boldsymbol{\pi} - \mathbf{q}\|_2$ es computable en $O(K \log K)$ mediante proyección sobre el símplex con cotas por coordenada.

**Nota:** Cuando $l_k = u_k = 1/K$, $SBC_{\text{proj}}$ colapsa exactamente a $SBC_W$.

#### Penalización Suave (para restricciones blandas)

Para escenarios donde las restricciones pueden violarse con costo, se define:

$$\Omega(\boldsymbol{\pi}, K) = \exp\!\left(-\lambda \sum_{k=1}^{K} v_k\right)$$

donde $v_k = \max(0, l_k - \pi_k) + \max(0, \pi_k - u_k)$ es la violación del cluster $k$, y $\lambda > 0$ controla la dureza de la penalización. Se usa $SBC_\Omega = \Omega(\boldsymbol{\pi}, K) \in (0, 1]$.

### 2.3 Confianza Muestral: $SCW$ (inspirado en JMDS)

Adapta el componente $LPG$ de JMDS al contexto de clustering con $K$ componentes GMM.

Para cada muestra $x_i$, se calcula la brecha de log-probabilidad:

$$LPG_K(x_i) = \log p(x_i \mid \text{comp}_{z^*_i}) - \log p(x_i \mid \text{comp}_{z^{**}_i})$$

donde:
- $z^*_i = \arg\max_k \log p(x_i \mid \text{comp}_k)$ es la componente primaria (asignación más probable)
- $z^{**}_i = \arg\max_{k \neq z^*_i} \log p(x_i \mid \text{comp}_k)$ es la componente secundaria

Un $LPG_K(x_i)$ alto indica que $x_i$ es claramente asignable a un cluster. Un $LPG_K(x_i)$ bajo (cercano a 0) indica muestra fronteriza que será **forzosamente reasignada** por el algoritmo de clustering con restricción de tamaño, degradando la calidad del agrupamiento.

La **Fracción de Masa en Frontera** (Boundary Mass Fraction) es:

$$BMF(D_s, K, \tau) = \frac{1}{n}\sum_{i=1}^{n} \mathbf{1}[LPG_K(x_i) < \tau]$$

donde $\tau$ es un umbral calibrado (e.g., percentil 25 de la distribución de $LPG_K$ sobre datasets de referencia, o $\tau = 0.5\,\hat{\sigma}_{LPG}$ donde $\hat{\sigma}_{LPG}$ es la desviación estándar empírica).

La confianza muestral global es:

$$SCW(D_s, K) = 1 - BMF(D_s, K, \tau) \in [0, 1]$$

#### Incorporación del Componente MPPL (opcional)

Si se dispone de un modelo fuente $\hat{p}_\theta$ entrenado en $D_s$ (e.g., un clasificador), se puede extender multiplicando por la confianza del modelo sobre la pseudo-etiqueta GMM:

$$MPPL(x_i) = \hat{p}_\theta(\hat{y}_i \mid x_i), \quad \hat{y}_i = z^*_i$$

$$SCW_{\text{JMDS}}(D_s, K) = SCW(D_s, K) \cdot \frac{1}{n}\sum_{i=1}^{n} MPPL(x_i)$$

**Recomendación:** Usar $SCW$ basado solo en $BMF$ como versión principal (métrica puramente estructural, sin acceso al modelo). Incorporar $MPPL$ como ablación experimental para cuantificar su contribución adicional.

---

## 3. Camino 1 — Fusión Multiplicativa

### Formulación

$$\boxed{\text{ITCR}_1(D_s, K) = CH_A(D_s, K) \cdot SBC_W(D_s, K) \cdot SCW(D_s, K)}$$

### Propiedades

El producto garantiza que si cualquier componente colapsa a 0, el índice global también colapsa. Esto es una propiedad **deseable y deliberada**: un dataset con pésimo balance de clusters no debe considerarse transferible sin importar su separabilidad, y viceversa.

Formalmente, $\text{ITCR}_1 \in [0,1]$ y es:
- **Monótono en $CH_A$:** Mayor separabilidad implica mayor transferibilidad, ceteris paribus.
- **Monótono en $SBC_W$:** Mayor equilibrio natural implica mayor transferibilidad.
- **Monótono en $SCW$:** Menor fracción de muestras fronterizas implica mayor transferibilidad.

### Ventajas

- Sin parámetros libres (no requiere calibración, solo el umbral $\tau$ de $SCW$).
- Alta interpretabilidad: descomposición exacta de la métrica en tres factores auditables.
- Ideal como **baseline** en la evaluación experimental.
- Reproducible y directamente comparable entre trabajos.

### Limitaciones

- Asume que los tres componentes tienen igual peso, lo que puede no ser óptimo para todos los escenarios.
- El producto es muy sensible a valores pequeños de cualquier componente (efecto "cuello de botella").

---

## 4. Camino 2 — Fusión Log-lineal Ponderada

### Formulación

$$\boxed{\text{ITCR}_2(D_s, K) = \exp\!\Big(\alpha \cdot \log CH_A + \beta \cdot \log SBC_W + \gamma \cdot \log SCW\Big)}$$

sujeto a: $\alpha + \beta + \gamma = 1$, con $\alpha, \beta, \gamma > 0$.

Equivalentemente: $\text{ITCR}_2 = CH_A^\alpha \cdot SBC_W^\beta \cdot SCW^\gamma$, que es la **media geométrica ponderada** de los tres componentes.

Cuando $\alpha = \beta = \gamma = 1/3$, se recupera $\text{ITCR}_1^{1/3}$ (equivalente en ordenación).

### Calibración de los Pesos

Los pesos $(\alpha, \beta, \gamma)$ se aprenden como un **meta-regresor** sobre un conjunto de datasets benchmark $\mathcal{B} = \{(D_s^{(j)}, K^{(j)}, q^{(j)})\}$ donde $q^{(j)} \in [0,1]$ es la calidad real del clustering con restricciones (medida con un índice de validación externo como Adjusted Rand Index o NMI):

$$(\hat{\alpha}, \hat{\beta}, \hat{\gamma}) = \arg\min_{\alpha+\beta+\gamma=1,\; \alpha,\beta,\gamma>0} \sum_j \left(\text{ITCR}_2(D_s^{(j)}, K^{(j)}) - q^{(j)}\right)^2$$

Este es un problema de optimización en el símplex 2D, resoluble con descenso de gradiente proyectado o mediante la parametrización $\alpha = \text{softmax}(\tilde{\alpha})_1$, etc.

### Interpretación de los Pesos

Si la calibración empírica produce $\hat{\beta} \gg \hat{\alpha}, \hat{\gamma}$, ello constituye **evidencia experimental** de que el balance de tamaño es el factor dominante para predecir la calidad del clustering con restricciones. Este análisis de sensibilidad puede ser una contribución experimental clave de la tesis.

### Ventajas

- Flexible: puede adaptarse a distintos escenarios (alta dimensionalidad, pocas muestras, restricciones muy estrictas).
- Los pesos son interpretables como importancias relativas de separabilidad, balance y confianza.
- Permite ablaciones: fijar $\gamma = 0$ para evaluar la contribución de $SCW$, etc.

### Limitaciones

- Requiere un benchmark de calibración con ground truth de calidad de clustering.
- Introduce riesgo de sobreajuste si $|\mathcal{B}|$ es pequeño.
- La interpretación es menos directa que el Camino 1.

---

## 5. Camino 3 — Extensión Axiomática de CLM (más original)

### Motivación

Este es el camino de mayor rigor teórico y originalidad. Se propone extender formalmente el framework CLM con un **quinto axioma** que incorpora la restricción de tamaño dentro del mismo proceso de ajuste logístico, en lugar de multiplicarla como un factor externo.

### Axioma A5 — Invariancia al Balance de Partición

> **A5 (Size-Constraint Invariance):** Para dos datasets $D$ y $D'$ con la misma estructura de separabilidad ($CH_K(D) = CH_K(D')$), la métrica debe asignar un valor mayor a aquel cuya distribución de pesos GMM $\boldsymbol{\pi}$ esté más cerca de la distribución objetivo $\mathbf{u}$. Formalmente: $W_1(\boldsymbol{\pi}_D, \mathbf{u}) < W_1(\boldsymbol{\pi}_{D'}, \mathbf{u}) \Rightarrow \text{ITCR}(D, K) > \text{ITCR}(D', K)$.

### Formulación

Se redefine el ajuste logístico de CLM incorporando el balance directamente como covariable adicional:

$$\boxed{CH_{A,K}(D_s) = \sigma\!\left(\beta_0 + \beta_1 \cdot \log(CH_K) + \beta_2 \cdot \log\!\left(1 - W_1(\boldsymbol{\pi}_K, \mathbf{u})\right)\right)}$$

donde:
- $CH_K = \frac{B_K/(K-1)}{W_K/(n-K)}$ es el índice Calinski-Harabasz crudo sobre el GMM con $K$ componentes
- $\beta_0, \beta_1, \beta_2$ son parámetros calibrados sobre datasets benchmark
- El término $\log(1 - W_1(\boldsymbol{\pi}_K, \mathbf{u}))$ actúa como penalización: cuando $\boldsymbol{\pi}_K = \mathbf{u}$, $W_1 = 0$ y el término vale 0 (penalización nula); cuando $\boldsymbol{\pi}_K$ está lejos de $\mathbf{u}$, $W_1 \to 1 - 1/K$ y el término $\to -\infty$

En este camino: $\text{ITCR}_3 = CH_{A,K}(D_s)$.

### Propiedades Axiomáticas

El $\text{ITCR}_3$ hereda A1–A4 de CLM (la función logística y el ratio $CH_K$ preservan todos los axiomas originales) y además satisface A5 por construcción, dado que $W_1(\boldsymbol{\pi}_K, \mathbf{u})$ es monótono en el desequilibrio y aparece con signo negativo dentro de $\sigma$.

### Calibración

Los parámetros $(\beta_0, \beta_1, \beta_2)$ se calibran mediante el mismo protocolo de CLM original: regresión logística sobre un conjunto de comparaciones pareadas $(D^{(j)}, D^{(j')})$ donde se conoce cuál dataset produce mejor clustering con restricciones. El clasificador binario aprende cuándo $CH_{A,K}(D^{(j)}) > CH_{A,K}(D^{(j')})$ es consistente con la preferencia humana/experimental.

### Ventajas

- El aporte teórico es más sólido: se extiende un framework axiomático existente con un nuevo axioma demostrable.
- La penalización de balance está "horneada" en la métrica, no es un multiplicador externo.
- Unifica en una sola función logística todos los factores, lo que simplifica el análisis de propiedades.

### Limitaciones

- Mayor costo de calibración: requiere tripletes $(\beta_0, \beta_1, \beta_2)$ en lugar de solo $(\beta_0, \beta_1)$.
- Menos modular que los Caminos 1 y 2: no permite ablaciones individuales de separabilidad vs. balance.

---

## 6. Selección de Subespacios (Feature Selection)

La métrica $\text{ITCR}$ se convierte en criterio de selección de features. Para un subconjunto $S \subseteq \{1, \ldots, d\}$ de tamaño $m$:

$$S^* = \arg\max_{S:\,|S|=m} \; \text{ITCR}(D_s^{(S)}, K)$$

### 6.1 ITCR-FS Greedy (Forward Selection)

**Algoritmo:**
1. Inicializar $S = \emptyset$, $\text{score}_{\text{prev}} = 0$
2. Para $t = 1, \ldots, m$:
   - $j^* = \arg\max_{j \notin S}\; \text{ITCR}(D_s^{(S \cup \{j\})}, K)$
   - Si $\text{ITCR}(D_s^{(S \cup \{j^*\})}, K) > \text{score}_{\text{prev}}$: $S \leftarrow S \cup \{j^*\}$
   - Else: detener (criterio de parada por ganancia marginal)
3. Devolver $S$

**Complejidad:** $O(m \cdot d)$ evaluaciones de $\text{ITCR}$, cada una requiriendo ajuste de un GMM con $K$ componentes: $O(n \cdot K \cdot d \cdot T_{EM})$ donde $T_{EM}$ es el número de iteraciones EM.

**Propiedad clave:** La ganancia marginal de $SBC_W$ guía la selección hacia features bajo los cuales los clusters tienen tamaños naturalmente balanceados, lo que es exactamente el objetivo del clustering con restricciones.

### 6.2 ITCR-FS Filtro (Univariado)

Se define la importancia de la feature $j$ como:

$$\text{Imp}(j) = \text{ITCR}(D_s^{(\{j\})}, K) \in [0, 1]$$

evaluada sobre el dataset univariado $D_s^{(\{j\})}$. Para capturar interacciones, se puede usar la ganancia de información mutual entre la feature $j$ y la asignación GMM $\hat{\mathbf{z}}$ en el espacio completo:

$$\text{Imp}_{MI}(j) = I(x^{(j)};\, \hat{\mathbf{z}}) = H(\hat{\mathbf{z}}) - H(\hat{\mathbf{z}} \mid x^{(j)})$$

Se seleccionan las top-$m$ features por $\text{Imp}(j)$ o $\text{Imp}_{MI}(j)$.

**Complejidad:** $O(d)$ evaluaciones de $\text{ITCR}$ univariadas — muy eficiente para $d$ grande.

### 6.3 ITCR-FS Embedded (Máscara Continua, Gradient-Based)

Se parametriza una máscara continua $\mathbf{w} \in [0,1]^d$ y se optimiza:

$$\max_{\mathbf{w} \in [0,1]^d} \; \text{ITCR}(D_s^{(\mathbf{w})}, K) - \lambda\|\mathbf{w}\|_1$$

donde $D_s^{(\mathbf{w})}$ es el dataset con features escaladas por $\mathbf{w}$ (i.e., $x_i^{(\mathbf{w})} = x_i \odot \mathbf{w}$).

El gradiente $\nabla_{\mathbf{w}} \text{ITCR}$ puede aproximarse por:
- **Diferencias finitas:** $\frac{\partial \text{ITCR}}{\partial w_j} \approx \frac{\text{ITCR}(\mathbf{w} + \epsilon \mathbf{e}_j) - \text{ITCR}(\mathbf{w} - \epsilon \mathbf{e}_j)}{2\epsilon}$ — costoso pero exacto.
- **Relajación del GMM:** Expresar los parámetros GMM como funciones diferenciables de $\mathbf{w}$ mediante el algoritmo EM suavizado (soft-EM), lo que permite backpropagation.

Al converger, umbralar $\mathbf{w}$ con un percentil produce el subespacio seleccionado: $S^* = \{j : w_j > w_{\text{threshold}}\}$.

### 6.4 Intuición sobre la Selección

Los tres componentes de $\text{ITCR}$ tienen sensibilidades distintas respecto a los features:

| Componente | Favorece features que... |
|---|---|
| $CH_A$ | Maximizan la separación inter-cluster |
| $SBC_W$ | Hacen que los clusters naturales tengan tamaños similares |
| $SCW$ | Reducen la masa de muestras en las fronteras de decisión |

La selección conjunta equilibra estos tres objetivos. Este **trade-off** es la hipótesis central de la tesis: los features que optimizan separabilidad pura no son necesariamente los que mejor se adaptan a clustering con restricciones de tamaño.

---

## 7. Comparación de los Tres Caminos

| Propiedad | Camino 1 (Multiplicativo) | Camino 2 (Log-lineal) | Camino 3 (Axiomático) |
|---|---|---|---|
| Axiomas CLM heredados | A1–A4 | A1–A4 | A1–A5 |
| Parámetros libres | 0 (+$\tau$) | 3 ($\alpha,\beta,\gamma$) | 3 ($\beta_0,\beta_1,\beta_2$) |
| Requiere calibración | No | Sí (benchmark) | Sí (benchmark) |
| Interpretabilidad | Alta | Media–Alta | Alta |
| Modularidad (ablaciones) | Alta | Alta | Baja |
| Rigor teórico | Medio | Medio | Alto |
| Originalidad | Media | Media | Alta |
| Uso recomendado | Baseline, reproducibilidad | Análisis de sensibilidad | Contribución teórica principal |

---

## 8. Protocolo Experimental Propuesto

### 8.1 Datasets Benchmark

Seleccionar un conjunto $\mathcal{B}$ de datasets etiquetados con propiedades variadas:
- Datasets con clases naturalmente balanceadas (UCI: Iris, Wine, Breast Cancer)
- Datasets con clases desbalanceadas (MNIST, CIFAR-10 subconjunto)
- Datasets de alta dimensionalidad (embeddings de texto, features de imagen)
- Datasets con clases con solapamiento

Para cada dataset y cada $K \in \{C, C/2, 2C\}$, ejecutar un algoritmo de clustering con restricciones de tamaño (e.g., Balanced K-Means, K-Medoids con restricciones) y medir calidad con ARI, NMI y Silhouette.

### 8.2 Evaluación de la Métrica

1. **Correlación de rango:** ¿Correlaciona $\text{ITCR}$ con la calidad real del clustering? Usar coeficiente de Spearman $\rho$.
2. **Calibración del Camino 2:** Aprender $(\alpha, \beta, \gamma)$ con validación cruzada sobre $\mathcal{B}$.
3. **Ablación:** Evaluar $CH_A$ solo, $CH_A \cdot SBC_W$, y $\text{ITCR}_1$ completo para cuantificar la contribución de cada componente.
4. **Feature selection:** Comparar clustering con features seleccionados por ITCR-FS vs. PCA, mRMR, ANOVA-F sobre los mismos datasets.

### 8.3 Hipótesis Verificables

- **H1:** $\text{ITCR}$ tiene mayor correlación de Spearman con la calidad del clustering con restricciones que $CH_A$ solo.
- **H2:** $SBC_W$ contribuye significativamente (el coeficiente $\hat{\beta}$ en el Camino 2 es estadísticamente significativo).
- **H3:** La selección de features por ITCR-FS produce mejor clustering con restricciones que PCA de igual rango.
- **H4:** El Camino 3 ($\text{ITCR}_3$) tiene igual o mayor correlación que los Caminos 1 y 2, con la ventaja adicional de satisfacer A5 axiomáticamente.

---

## 9. Recomendación Final

Para la estructura de la tesis, se sugiere el siguiente orden de desarrollo:

1. **Presentar el Camino 1** como métrica principal: sin parámetros libres, interpretable, y verificable sin benchmark de calibración. Úsalo como el resultado central de la tesis.

2. **Presentar el Camino 2** en la sección experimental como análisis de sensibilidad: ¿qué peso aprende el modelo para cada componente? Si $\hat{\beta} \gg \hat{\alpha}, \hat{\gamma}$, esto confirma la hipótesis de que el balance es el factor dominante en clustering con restricciones.

3. **Presentar el Camino 3** como extensión teórica en un capítulo de contribuciones avanzadas: la demostración formal de que A5 es independiente de A1–A4 y que $\text{ITCR}_3$ es el único índice CLM-compatible que satisface los cinco axiomas.

4. **Presentar ITCR-FS** como aplicación práctica: la métrica no solo evalúa datasets, sino que también guía la selección de representaciones óptimas para el problema de clustering restringido.

---
