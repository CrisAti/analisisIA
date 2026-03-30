# Diseño de una Métrica de Transferencia para la Utilización de Conjuntos de Datos de Clasificación en Tareas de Clustering

> **TCSC — Transfer Compatibility for Size-Constrained Clustering**
> Documento de trabajo para tesis de maestría/doctorado.

---

## Tabla de Contenidos

1. [Motivación y Problema](#1-motivación-y-problema)
2. [Fundamentos Teóricos](#2-fundamentos-teóricos)
   - 2.1 CLM y Axiomas Cruzados
   - 2.2 JMDS y Confianza Local
   - 2.3 Clustering con Restricciones de Tamaño
3. [Nuevo Axioma A5: Compatibilidad de Tamaño](#3-nuevo-axioma-a5-compatibilidad-de-tamaño)
4. [Formulación de la Métrica TCSC](#4-formulación-de-la-métrica-tcsc)
   - 4.1 Componente Estructural Global con Confianza Local
   - 4.2 Puntuación de Confianza por Muestra (Adaptación de JMDS)
   - 4.3 Penalización de Compatibilidad de Tamaño
   - 4.4 Integración Coherente
5. [Extensión del Marco Axiomático](#5-extensión-del-marco-axiomático)
6. [Modelado de la Compatibilidad de Tamaño](#6-modelado-de-la-compatibilidad-de-tamaño)
7. [Auto-calibración y Parámetros](#7-auto-calibración-y-parámetros)
8. [Selección de Subespacios y Features](#8-selección-de-subespacios-y-features)
9. [Diseño Experimental](#9-diseño-experimental)
10. [Ventajas, Limitaciones e Implicaciones](#10-ventajas-limitaciones-e-implicaciones)
11. [Resumen Matemático](#11-resumen-matemático)

---

## 1. Motivación y Problema

### 1.1 El Problema Central

Dado un conjunto de datos etiquetado $D = \{(x_i, y_i)\}_{i=1}^n$ con $x_i \in \mathbb{R}^d$ e $y_i \in \{1, \ldots, K\}$, queremos determinar **qué tan apto es $D$ para ser utilizado como punto de partida en una tarea de clustering bajo restricciones de tamaño**.

Las restricciones de tamaño son de la forma general:

$$\Pi^* = \left\{ \pi = (\pi_1, \ldots, \pi_K) \;\middle|\; \frac{L_k}{n} \leq \pi_k \leq \frac{U_k}{n}, \quad \sum_{k=1}^K \pi_k = 1 \right\}$$

donde $L_k, U_k$ son cotas inferiores y superiores para el cluster $k$, **sin asumir uniformidad** ($\pi_k \neq 1/K$ en general). La región $\Pi^*$ puede también representar proporciones objetivo no uniformes $\pi^* = (\pi_1^*, \ldots, \pi_K^*)$.

### 1.2 Por qué Necesitamos una Nueva Métrica

Las métricas existentes fallan por dos razones distintas:

**Problema 1 — Las métricas de validación interna (IVM) no son transferibles entre datasets.** El índice Calinski-Harabasz (CH), Silhouette (SC) y similares miden la calidad estructural *dentro* de un dataset, pero sus valores no son comparables *entre* datasets de distintas dimensiones, cardinalidades o números de clases. Jeon et al. (2025) demostraron que $\text{CH}(D_1) > \text{CH}(D_2)$ no implica que $D_1$ tenga mejor CLM que $D_2$.

**Problema 2 — Ninguna métrica existente considera las restricciones de tamaño objetivo.** Incluso los IVM ajustados (IVM$_A$) de Jeon et al. (2025) evalúan la calidad estructural sin considerar si la distribución natural de los clusters en $D$ es compatible con $\Pi^*$. Un dataset con CLM excelente pero con clusters naturalmente muy desbalanceados será un mal punto de partida para un clustering que requiere proporciones específicas.

### 1.3 El Concepto de Transferibilidad

Definimos la **transferibilidad** de un dataset $D$ hacia una tarea de clustering con restricciones $\Pi^*$ como una función $\mathcal{T}: \mathcal{D} \times \mathcal{P} \to [0,1]$ que satisface:

- **Alta transferibilidad** cuando las clases de $D$ forman clusters bien separados Y la distribución natural de esos clusters es compatible con $\Pi^*$.
- **Baja transferibilidad** cuando las clases se solapan, están mal definidas, o cuando la distribución natural de tamaños es incompatible con las restricciones.

---

## 2. Fundamentos Teóricos

### 2.1 CLM: Cluster-Label Matching

El framework CLM (Jeon et al., 2025) evalúa la validez intrínseca de un dataset etiquetado para clustering. Propone **cuatro axiomas cruzados** que una medida $f$ debe satisfacer para comparar CLM entre datasets:

| Axioma | Nombre | Descripción |
|--------|--------|-------------|
| **A1** | Data-Cardinality Invariance | $f(C, X, \delta) = f(C^\alpha, X^\alpha, \delta)$ — invariante a submuestreo proporcional |
| **A2** | Shift Invariance | $f(C, X, \delta) = f(C, X, \delta + \beta)$ — invariante a desplazamiento global de distancias |
| **A3** | Class-Cardinality Invariance | $f(C, X, \delta) = \text{agg}_{S \subseteq C, |S|=2} f(S, X, \delta)$ — agregación por pares de clases |
| **A4** | Range Invariance | $\min_C f = 0$, $\max_C f = 1$ — rango normalizado y comparable |

La medida ajustada **CHA** (Calinski-Harabasz Ajustado) satisface todos estos axiomas y se construye aplicando protocolos T1–T4 sucesivamente:

$$\text{CHA}(C, X, d^2) = \frac{1}{\binom{|C|}{2}} \sum_{S \subseteq C, |S|=2} \text{CH}_5(S, X, d^2)$$

donde $\text{CH}_5$ es la versión normalizada, shift-invariante y con rango acotado del índice CH original.

**Lo que CLM mide:** qué tan bien las etiquetas de clase se alinean con la estructura de clusters natural de los datos.

**Lo que CLM NO mide:** (i) la confianza local de cada muestra en su asignación, y (ii) la compatibilidad de la distribución de tamaños con $\Pi^*$.

### 2.2 JMDS: Joint Model-Data Structure Score

El score JMDS (Lee et al., 2022) fue diseñado para SFUDA, pero sus principios son transferibles. Combina dos fuentes de conocimiento:

**Log-Probability Gap (LPG)** — confianza basada en la estructura de datos:

$$\text{MINGAP}(x_i) = \min_{a \neq \hat{y}_i} \left\{ \log p_{\text{data}}(x_i)_{\hat{y}_i} - \log p_{\text{data}}(x_i)_a \right\}$$

$$\text{LPG}(x_i) = \frac{\text{MINGAP}(x_i)}{\max_j \text{MINGAP}(x_j)}$$

**Model Probability of Pseudo-Label (MPPL)** — confianza del modelo:

$$\text{MPPL}(x_i) = p_M(x_i)_{\hat{y}_i}$$

**Score JMDS:** producto de ambos (enfatiza muestras confiables en ambas dimensiones):

$$\text{JMDS}(x_i) = \text{LPG}(x_i) \cdot \text{MPPL}(x_i) \in [0,1]$$

**Lo que JMDS captura:** confianza *por muestra*, considerando tanto la geometría local del espacio de features como la certeza del modelo de clasificación.

**Adaptación a nuestro contexto:** En lugar de usarlo para adaptar un modelo a un dominio target, lo usamos para **ponderar la contribución de cada muestra** a la evaluación estructural del dataset.

### 2.3 Clustering con Restricciones de Tamaño

Un clustering con restricciones de tamaño busca:

$$\arg\min_{\{C_k\}_{k=1}^K} \sum_{k=1}^K \sum_{x \in C_k} d(x, \mu_k)^2 \quad \text{s.t.} \quad L_k \leq |C_k| \leq U_k \;\; \forall k$$

La distribución de proporciones **natural** del dataset (inducida por GMM o por las etiquetas) se define como:

$$\hat{\pi} = (\hat{\pi}_1, \ldots, \hat{\pi}_K), \quad \hat{\pi}_k = \frac{|C_k|}{n}$$

La **región factible de proporciones** es:

$$\Pi^* = \left\{ \pi \in \Delta^{K-1} \;\middle|\; \frac{L_k}{n} \leq \pi_k \leq \frac{U_k}{n} \;\forall k \right\}$$

donde $\Delta^{K-1}$ es el simplex estándar $(K-1)$-dimensional.

---

## 3. Nuevo Axioma A5: Compatibilidad de Tamaño

### 3.1 Formulación Formal

> **Axioma A5 — Size-Constraint Compatibility (SCC):**
>
> Sea $f: \mathcal{C} \times \mathcal{X} \times \mathcal{D}_\Pi \to [0,1]$ una métrica de transferibilidad que opera sobre pares $(D, \Pi^*)$ de dataset y región factible de proporciones.
>
> $f$ satisface **compatibilidad de tamaño** si, para todo par de datasets $(D_1, D_2)$ con igual calidad estructural $\text{CLM}(D_1) = \text{CLM}(D_2)$, y para toda región factible $\Pi^*$:
>
> $$d_{\Pi}(\hat{\pi}^{(1)}, \Pi^*) \leq d_{\Pi}(\hat{\pi}^{(2)}, \Pi^*) \implies f(D_1, \Pi^*) \geq f(D_2, \Pi^*)$$
>
> donde $\hat{\pi}^{(m)}$ es la distribución natural de proporciones de $D_m$ y $d_{\Pi}(\hat{\pi}, \Pi^*) = \min_{\pi \in \Pi^*} D(\hat{\pi} \| \pi)$ es la distancia a la región factible bajo alguna divergencia $D$.

### 3.2 Por qué A5 No Está Cubierto por A1–A4

**A1 (Cardinality Invariance)** exige que el submuestreo proporcional no cambie el score. Esto concierne al tamaño absoluto del dataset ($n$), no a la *distribución relativa* entre clusters ni a su compatibilidad con $\Pi^*$.

**A2 (Shift Invariance)** opera sobre distancias en el espacio de features. La distribución $\hat{\pi}$ es una propiedad del espacio de etiquetas, no del espacio métrico $\delta$.

**A3 (Class-Cardinality Invariance)** exige que la métrica no dependa del número total de clases $K$ mediante agregación por pares. Pero no contempla que las proporciones relativas de esas clases deban ser compatibles con un objetivo externo $\Pi^*$.

**A4 (Range Invariance)** normaliza los valores de la métrica. No impone ninguna relación entre la métrica y las restricciones de tamaño.

**Conclusión:** A1–A4 forman un sistema cerrado sobre propiedades del espacio métrico y la cardinalidad del dataset. A5 introduce una noción *externa* — la compatibilidad con un objetivo de proporciones $\Pi^*$ — que ningún axioma previo puede capturar porque ninguno hace referencia a $\Pi^*$.

### 3.3 Consecuencia Formal

A5 implica que la métrica TCSC debe poder descomponerse (o factorizarse) en al menos dos términos: uno que capture la calidad estructural (invariante a $\Pi^*$) y otro que capture la compatibilidad de tamaño (sensible a $\Pi^*$). La integración debe ser **monótona** en ambos términos.

---

## 4. Formulación de la Métrica TCSC

La métrica TCSC se define como:

$$\boxed{\text{TCSC}(D, \Pi^*) = \Phi_{\text{struct}}^w(D) \cdot \Psi_{\text{size}}(D, \Pi^*)}$$

donde:
- $\Phi_{\text{struct}}^w(D) \in [0,1]$ — calidad estructural global ponderada por confianza local
- $\Psi_{\text{size}}(D, \Pi^*) \in [0,1]$ — compatibilidad de la distribución natural con las restricciones

**Justificación de la integración como producto:** El producto es la forma más natural de combinar dos condiciones **necesarias** para la transferibilidad. Si la estructura es perfecta ($\Phi = 1$) pero las proporciones son incompatibles ($\Psi = 0$), el dataset no es transferible — y viceversa. Matemáticamente, el producto implementa una AND-lógica suave: $\text{TCSC} = 1$ sólo cuando ambos componentes son máximos. Esta es una justificación más fuerte que una combinación convexa, que toleraría compensación entre componentes.

### 4.1 Componente Estructural Global con Confianza Local: $\Phi_{\text{struct}}^w$

**Idea:** Tomamos el CHA (Jeon et al., 2025) como base y reemplazamos el promedio uniforme sobre muestras por un promedio ponderado por la confianza local de cada muestra.

**Paso 1 — Estimación de la estructura por GMM:**

Ajustamos un GMM con $K$ componentes sobre las features $\{x_i\}$:

$$p_{\text{GMM}}(x) = \sum_{k=1}^K \hat{\pi}_k^{\text{GMM}} \cdot \mathcal{N}(x; \mu_k, \Sigma_k)$$

Esto nos da:
- La distribución natural de proporciones: $\hat{\pi}_k^{\text{GMM}} = \hat{\pi}_k$
- La probabilidad de pertenencia de cada muestra al cluster $k$: $\gamma_{ik} = p(z_i = k | x_i)$
- El pseudo-label $\hat{y}_i = \arg\max_k \gamma_{ik}$

**Paso 2 — Score de confianza local (adaptación de JMDS):**

Definimos el score de **confianza estructural** (análogo a LPG):

$$\text{CS}_{\text{struct}}(x_i) = \frac{\text{MINGAP}_{\text{GMM}}(x_i)}{\max_j \text{MINGAP}_{\text{GMM}}(x_j)}$$

$$\text{MINGAP}_{\text{GMM}}(x_i) = \min_{a \neq \hat{y}_i} \left\{ \log \gamma_{i\hat{y}_i} - \log \gamma_{ia} \right\}$$

Definimos el score de **confianza del clasificador** (análogo a MPPL):

Entrenamos un clasificador $g$ sobre $\{(x_i, y_i)\}$ (por ejemplo, un SVM lineal o kNN en el espacio de features). La confianza del clasificador es:

$$\text{CS}_{\text{model}}(x_i) = p_g(x_i)_{y_i}$$

donde $p_g(x_i)_{y_i}$ es la probabilidad (softmax o probabilidad calibrada) asignada a la clase verdadera $y_i$.

**Score de confianza local integrado (TCSC-conf):**

$$w_i = \text{CS}_{\text{struct}}(x_i) \cdot \text{CS}_{\text{model}}(x_i) \in [0,1]$$

**Paso 3 — CHA ponderado por confianza:**

Para cada par de clases $S = \{C_a, C_b\} \subseteq C$, redefinimos las estadísticas de CH con pesos:

Sea $w_i^{(S)}$ el peso de la muestra $i$ dentro del par $S$ (normalizado dentro del par). Definimos:

$$\mu_a^w = \frac{\sum_{i \in C_a} w_i x_i}{\sum_{i \in C_a} w_i}, \quad \mu^w = \frac{\sum_{i \in C_a \cup C_b} w_i x_i}{\sum_{i \in C_a \cup C_b} w_i}$$

El CH ponderado para el par $S$:

$$\text{CH}_w(S, X, d^2) = \frac{\sum_{k \in S} \left(\sum_{i \in C_k} w_i\right) d^2(\mu_k^w, \mu^w)}{\sum_{k \in S} \sum_{i \in C_k} w_i \cdot d^2(x_i, \mu_k^w)}$$

Aplicamos los protocolos T1–T4 de Jeon et al. (2025) a $\text{CH}_w$ para obtener $\text{CH}_w^5$, y finalmente:

$$\Phi_{\text{struct}}^w(D) = \frac{1}{\binom{K}{2}} \sum_{S \subseteq C, |S|=2} \text{CH}_w^5(S, X, d^2)$$

**Intuición:** Muestras en regiones ambiguas (bajo $w_i$) contribuyen menos al cálculo de separabilidad y compacidad. Esto hace que $\Phi_{\text{struct}}^w$ sea más conservador que CHA cuando hay regiones dudosas, y equivalente a CHA cuando todas las muestras tienen alta confianza.

**Propiedad:** $\Phi_{\text{struct}}^w$ preserva los axiomas A1–A4 porque:
- Los pesos $w_i$ son invariantes a reescalado de distancias (A2), ya que dependen de log-ratios de probabilidades GMM.
- La agregación por pares sigue satisfaciendo A3.
- La normalización T4 garantiza A4.
- T1 sigue siendo aplicable porque los pesos son estadísticas de población consistentes.

### 4.2 Score de Confianza por Muestra — Detalles

**Estimación de $p_g$:**

Si se usa un clasificador probabilístico (e.g., Naive Bayes, regresión logística, SVM con calibración de Platt), $p_g(x_i)_{y_i}$ es directo. Para maximizar generalización sin sobreajuste, se recomienda estimación por **cross-validation leave-one-out** (LOOCV):

$$\text{CS}_{\text{model}}(x_i) = p_{g_{-i}}(x_i)_{y_i}$$

donde $g_{-i}$ es el clasificador entrenado sin la muestra $i$.

**Alternativa sin clasificador (totalmente no supervisada):**

Si queremos evitar el entrenamiento de un clasificador, podemos usar únicamente:

$$w_i = \text{CS}_{\text{struct}}(x_i)$$

obteniendo una versión más simple pero igualmente válida. La componente de modelo puede aproximarse por la probabilidad marginal del GMM:

$$\text{CS}_{\text{model}}^{\text{GMM}}(x_i) = \frac{p_{\text{GMM}}(x_i | \mu_{y_i}, \Sigma_{y_i})}{p_{\text{GMM}}(x_i)}$$

que es la responsabilidad a posteriori del GMM para la clase verdadera $y_i$.

### 4.3 Penalización de Compatibilidad de Tamaño: $\Psi_{\text{size}}$

**Idea central:** medir qué tan lejos está la distribución natural $\hat{\pi}$ del dataset de la región factible $\Pi^*$.

**Distribución natural estimada:**

$$\hat{\pi}_k = \hat{\pi}_k^{\text{GMM}} = \frac{\sum_i \gamma_{ik}}{n} \quad (\text{o simplemente } |C_k|/n \text{ con etiquetas})$$

**Distancia proyectada a $\Pi^*$:**

Si $\Pi^*$ es un politopo convexo (definido por cotas superiores e inferiores), la proyección es un problema de programación cuadrática (QP):

$$\pi^* = \arg\min_{\pi \in \Pi^*} \|\hat{\pi} - \pi\|_2^2$$

La distancia de Wasserstein $W_1$ también es válida cuando las clases tienen un orden natural.

Definimos la distancia normalizada:

$$\delta_{\Pi}(\hat{\pi}, \Pi^*) = \frac{\|\hat{\pi} - \pi^*\|_2}{\sqrt{2}} \in [0, 1]$$

(normalizada por $\sqrt{2}$, el diámetro máximo posible del simplex bajo $\ell_2$).

La compatibilidad de tamaño es:

$$\Psi_{\text{size}}(D, \Pi^*) = 1 - \delta_{\Pi}(\hat{\pi}, \Pi^*)$$

**Caso $\Pi^*$ dado por proporciones objetivo $\pi^*$ (no una región):**

$$\Psi_{\text{size}}(D, \pi^*) = 1 - \frac{1}{\sqrt{2}} \cdot \|\hat{\pi} - \pi^*\|_2$$

**Caso $\Pi^*$ dado por divergencia KL:**

$$\delta_{\Pi}^{\text{KL}}(\hat{\pi}, \pi^*) = \frac{D_{\text{KL}}(\hat{\pi} \| \pi^*)}{\log K}$$

$$\Psi_{\text{size}}^{\text{KL}}(D, \pi^*) = e^{-D_{\text{KL}}(\hat{\pi} \| \pi^*)}$$

(forma exponencial, que es suave y tiene soporte $[0,1]$).

**Caso restricciones desiguales ($L_k \leq n\hat{\pi}_k \leq U_k$):**

$$\delta_{\Pi}(\hat{\pi}, \Pi^*) = \frac{\left\| \max\left(0, \frac{L}{n} - \hat{\pi}\right) + \max\left(0, \hat{\pi} - \frac{U}{n}\right) \right\|_1}{2}$$

Esta suma captura la **violación total** de cotas, normalizada. Cuando $\hat{\pi} \in \Pi^*$, $\delta_\Pi = 0$, y $\Psi_{\text{size}} = 1$.

### 4.4 Integración Coherente

La métrica final es:

$$\text{TCSC}(D, \Pi^*) = \Phi_{\text{struct}}^w(D) \cdot \Psi_{\text{size}}(D, \Pi^*)$$

**Propiedades de la integración:**

1. **Condición necesaria:** $\text{TCSC} = 1 \Leftrightarrow \Phi_{\text{struct}}^w = 1 \text{ y } \Psi_{\text{size}} = 1$. Ambas condiciones son necesarias para la transferibilidad perfecta.

2. **Monotonía:** $\text{TCSC}$ es monótona creciente en $\Phi_{\text{struct}}^w$ y en $\Psi_{\text{size}}$ (manteniendo el otro fijo).

3. **Desacoplamiento:** Si $\Pi^*$ es el simplex completo ($L_k = 0, U_k = n$), entonces $\Psi_{\text{size}} = 1$ y $\text{TCSC} = \Phi_{\text{struct}}^w$ (degeneración correcta al caso sin restricciones).

4. **Colapsibilidad:** Si $\Phi_{\text{struct}}^w = 0$ (clases sin estructura), $\text{TCSC} = 0$ independientemente de la compatibilidad de tamaños. Esto es correcto: un dataset sin estructura no puede transferirse a ningún clustering.

**Por qué el producto y no una combinación convexa:**

Una combinación convexa $\lambda \Phi + (1-\lambda) \Psi$ permitiría que una penalización de tamaño nula ($\Psi = 0$) sea compensada por una estructura perfecta ($\Phi = 1$). Pero en la práctica, si la distribución natural es completamente incompatible con $\Pi^*$, ninguna calidad estructural hace que el dataset sea transferible: el clustering restringido forzará asignaciones que violan la estructura natural. El producto evita esta compensación espuria.

---

## 5. Extensión del Marco Axiomático

### 5.1 Marco Completo (A1–A5)

Una métrica $f: \mathcal{D} \times \mathcal{P} \to [0,1]$ es una **métrica de transferibilidad válida** para clustering con restricciones de tamaño si y sólo si satisface:

- **A1** Data-Cardinality Invariance (heredado de CLM)
- **A2** Shift Invariance (heredado de CLM)
- **A3** Class-Cardinality Invariance (heredado de CLM)
- **A4** Range Invariance (heredado de CLM)
- **A5** Size-Constraint Compatibility (nuevo)

### 5.2 Verificación de que TCSC Satisface A5

**Proposición:** TCSC satisface A5.

**Demostración:** Sea $D_1, D_2$ con $\Phi_{\text{struct}}^w(D_1) = \Phi_{\text{struct}}^w(D_2) = \phi$ y sea $\delta_\Pi(\hat{\pi}^{(1)}, \Pi^*) \leq \delta_\Pi(\hat{\pi}^{(2)}, \Pi^*)$.

Entonces:
$$\Psi_{\text{size}}(D_1, \Pi^*) = 1 - \delta_\Pi(\hat{\pi}^{(1)}, \Pi^*) \geq 1 - \delta_\Pi(\hat{\pi}^{(2)}, \Pi^*) = \Psi_{\text{size}}(D_2, \Pi^*)$$

Por tanto:
$$\text{TCSC}(D_1, \Pi^*) = \phi \cdot \Psi_{\text{size}}(D_1, \Pi^*) \geq \phi \cdot \Psi_{\text{size}}(D_2, \Pi^*) = \text{TCSC}(D_2, \Pi^*)$$

$\square$

### 5.3 Consistencia del Sistema Extendido

El sistema A1–A5 es **consistente** porque TCSC lo satisface (por construcción). Es **más completo** que A1–A4 porque los IVM$_A$ estándar no satisfacen A5 (no dependen de $\Pi^*$). Sin embargo, como toda axiomatización, no es **completo** en sentido estricto: podría existir otra función que satisfaga A1–A5 y no sea una métrica de transferibilidad válida.

---

## 6. Modelado de la Compatibilidad de Tamaño

Esta sección profundiza en los distintos modelos para $\Psi_{\text{size}}$, con justificaciones matemáticas y casos de uso.

### 6.1 Distancia Proyectada (recomendada para restricciones de caja)

**Cuándo usar:** Restricciones $L_k \leq n\pi_k \leq U_k$ (cotas absolutas o relativas).

**Formulación:** El problema de proyección es:

$$\pi^* = \arg\min_\pi \|\hat{\pi} - \pi\|_2^2 \quad \text{s.t.} \quad \frac{L_k}{n} \leq \pi_k \leq \frac{U_k}{n}, \sum_k \pi_k = 1$$

Este QP convexo se resuelve en $O(K \log K)$ mediante el algoritmo de proyección sobre el simplex con restricciones de caja (Duchi et al., 2008).

$$\Psi_{\text{proj}}(D, \Pi^*) = 1 - \frac{\|\hat{\pi} - \pi^*\|_2}{\sqrt{2}}$$

### 6.2 Divergencia KL Generalizada

**Cuándo usar:** Cuando $\Pi^*$ se expresa como una distribución objetivo $\pi^* = (\pi_1^*, \ldots, \pi_K^*)$ con interpretación probabilística.

**Formulación:**

$$\Psi_{\text{KL}}(D, \pi^*) = \exp\left(-D_{\text{KL}}(\hat{\pi} \| \pi^*)\right) = \exp\left(-\sum_k \hat{\pi}_k \log \frac{\hat{\pi}_k}{\pi_k^*}\right)$$

**Propiedad:** $\Psi_{\text{KL}} = 1$ ssi $\hat{\pi} = \pi^*$. El factor exponencial garantiza $\Psi \in [0,1]$.

**Variante simétrica (Jensen-Shannon):**

$$\Psi_{\text{JS}}(D, \pi^*) = 1 - \text{JSD}(\hat{\pi} \| \pi^*) = 1 - \frac{1}{2}\left[D_{\text{KL}}(\hat{\pi} \| m) + D_{\text{KL}}(\pi^* \| m)\right]$$

donde $m = (\hat{\pi} + \pi^*)/2$. La distancia Jensen-Shannon está acotada en $[0, \log 2]$, por lo que $\Psi_{\text{JS}} \in [1 - \log 2, 1] \approx [0.307, 1]$. Para normalización, dividir por $\log 2$.

### 6.3 Penalización Dura vs. Suave

**Penalización dura:** $\Psi_{\text{hard}} = \mathbb{1}[\hat{\pi} \in \Pi^*]$ — solo toma valores 0 o 1. No es diferenciable y produce una métrca con poca información gradiente. **No recomendada** para optimización.

**Penalización suave (propuesta):** Las formulaciones anteriores son continuas y derivables respecto a $\hat{\pi}$, lo que permite su uso como función objetivo en selección de features (Sección 8).

### 6.4 Robustez ante Estimación de $\hat{\pi}$

La distribución natural $\hat{\pi}$ se estima del GMM o de las frecuencias de etiqueta. Ambas son estimadores consistentes bajo las hipótesis del modelo. Para mayor robustez, se puede usar un estimador Bayesiano con prior de Dirichlet:

$$\hat{\pi}_k^{\text{Bayes}} = \frac{|C_k| + \alpha_k}{n + \sum_k \alpha_k}$$

donde $\alpha_k$ son hiperparámetros del prior. Con $\alpha_k = 1/K$ (prior uniforme suave), se obtiene un estimador de Laplace que evita proporciones exactamente cero.

---

## 7. Auto-calibración y Parámetros

### 7.1 Inventario de Parámetros

| Parámetro | Rol | Estrategia de calibración |
|-----------|-----|--------------------------|
| $K$ (número de clusters) | Definido por la tarea | Externo, dado por el usuario |
| GMM (covarianza) | Estimación de $\hat{\pi}$ y $\gamma_{ik}$ | EM estándar; sin parámetros libres adicionales |
| $k$ (tasa logística en T4-c) | Normalización del rango de CHA | Calibrado con percepciones humanas (Jeon et al., 2025) |
| Tipo de distancia en $\Psi_{\text{size}}$ | Selección de $\|\cdot\|$ o divergencia | Basado en la estructura de $\Pi^*$ |

### 7.2 El Único Parámetro Libre: $k$ en T4-c

El único parámetro que requiere calibración externa es la tasa de crecimiento logístico $k$ de CHA. Jeon et al. (2025) lo calibran con datos de percepción humana de separabilidad de clusters (Abbas et al., 2019), utilizando optimización Bayesiana para maximizar el $R^2$ entre scores humanos y CHA.

**Alternativa auto-calibrada:** Fijar $k$ de forma que la función logística mapee el percentil 50 de CH (sobre random partitions) al valor $0.5$:

$$k^* = \frac{\log(1/0.5 - 1)}{\text{median}_\pi[\text{CH}_3(C_\pi, X, d^2)] - 0}$$

Esto equivale a calibrar con la distribución nula del dataset mismo, sin datos externos. Es conceptualmente limpio y generaliza bien porque usa el propio dataset para definir el punto de referencia.

### 7.3 Formulación Sin Parámetros Libres (variante)

Si se reemplaza $\text{CH}_5$ (que usa T4-c con $k$) por la versión basada en Monte Carlo pura (T4-a):

$$\text{CH}_5^{\text{MC}}(C, X, d^2) = \frac{\text{CH}_4(C, X, d^2) - \hat{\text{CH}}_4^{\min}}{\text{CH}_4^{\max} - \hat{\text{CH}}_4^{\min}}$$

donde $\hat{\text{CH}}_4^{\min} = 1/2$ (aproximación analítica de Jeon et al.), se elimina $k$ completamente. La pérdida de calibración es marginal en práctica (Jeon et al., 2025, Sección V-A).

---

## 8. Selección de Subespacios y Features

### 8.1 Formulación del Problema

Dado un dataset $D$ con features en $\mathbb{R}^d$, buscamos el subconjunto de features $\mathbf{w} \in \{0,1\}^d$ que maximiza TCSC:

$$\mathbf{w}^* = \arg\max_{\mathbf{w} \in \{0,1\}^d} \text{TCSC}(D_\mathbf{w}, \Pi^*)$$

donde $D_\mathbf{w}$ es $D$ restringido a las features seleccionadas por $\mathbf{w}$.

### 8.2 Por qué TCSC es Adecuado para Selección de Features

TCSC tiene dos propiedades clave para la selección de features:

1. **Sensibilidad a la geometría local:** $\Phi_{\text{struct}}^w$ captura cómo la separabilidad de clusters cambia al agregar o quitar features. Features que reducen el solapamiento aumentan $\Phi_{\text{struct}}^w$.

2. **Sensibilidad a la distribución de tamaños:** $\Psi_{\text{size}}$ penaliza features que distorsionan la distribución natural hacia proporciones incompatibles con $\Pi^*$. Esto es único de TCSC — ninguna métrica estándar hace esto.

### 8.3 Algoritmo de Selección Greedy

```
Entrada: D, Π*, presupuesto de features m
Salida: subconjunto de features W* de tamaño m

1. Inicializar W = {} (conjunto vacío)
2. Para t = 1, ..., m:
   a. Para cada feature j ∉ W:
      - Calcular TCSC(D_{W ∪ {j}}, Π*)
   b. Seleccionar j* = argmax_j TCSC(D_{W ∪ {j}}, Π*)
   c. W ← W ∪ {j*}
3. Retornar W
```

**Complejidad:** $O(m \cdot d \cdot n K^2)$ — lineal en $n$ y $K^2$ (heredado de CHA).

### 8.4 Selección Continua por Relajación

Para datasets de alta dimensionalidad (e.g., embeddings de MNIST con 784 dimensiones), la búsqueda discreta es costosa. Relajamos $\mathbf{w} \in [0,1]^d$ y minimizamos:

$$\mathcal{L}(\mathbf{w}) = -\text{TCSC}(D_\mathbf{w}, \Pi^*) + \lambda \|\mathbf{w}\|_1$$

El término $\ell_1$ promueve esparsidad. $\Phi_{\text{struct}}^w$ es diferenciable respecto a $\mathbf{w}$ (mediante la cadena de derivación a través del GMM y CHA ponderado). $\Psi_{\text{size}}$ también es diferenciable porque $\hat{\pi}$ depende suavemente de $\mathbf{w}$ a través de la responsabilidad GMM.

**Valor de $\lambda$:** Se puede seleccionar por la regla SURE o fijarlo en $\lambda = 1/d$, dando igual peso a la regularización y al objetivo.

### 8.5 Interpretación en el Contexto de Size-Constrained Clustering

La selección de features con TCSC tiene una interpretación específica para clustering con restricciones de tamaño:

- Features con alto peso en $\mathbf{w}^*$ son aquellas que **simultáneamente** hacen los clusters más separables Y más compatibles con $\Pi^*$.
- Features que mejoran la estructura pero desvían $\hat{\pi}$ de $\Pi^*$ serán penalizadas, aunque mejoren la calidad de clustering sin restricciones.
- Esto produce una selección de features **específica para la tarea restringida**, no genérica.

---

## 9. Diseño Experimental

### 9.1 Datasets de Computer Vision

| Dataset | $n$ | $d_{\text{raw}}$ | $K$ | Descripción |
|---------|-----|-----------------|-----|-------------|
| MNIST | 70,000 | 784 | 10 | Dígitos manuscritos |
| Fashion-MNIST | 70,000 | 784 | 10 | Prendas de ropa |
| CIFAR-10 | 60,000 | 3,072 | 10 | Imágenes naturales |
| CIFAR-100 | 60,000 | 3,072 | 100 | Imágenes naturales (100 clases) |
| STL-10 | 13,000 | 27,648 | 10 | Imágenes de alta resolución |

Para todos, se usarán embeddings de ResNet-50 pre-entrenado en ImageNet (dimensión 2,048 o reducida a 128 con PCA/UMAP), no los pixels raw.

### 9.2 Configuraciones de Restricciones de Tamaño

Se proponen cinco configuraciones de $\Pi^*$ para capturar distintos niveles de desbalance y factibilidad:

**C1 — Sin restricciones (baseline):** $\Pi^* = \Delta^{K-1}$. $\Psi_{\text{size}} = 1$ siempre.

**C2 — Uniforme (balanced):** $\pi_k^* = 1/K \;\forall k$. Caso estándar de balanced k-means.

**C3 — Desbalance moderado (log-uniforme):** $\pi_k^* \propto \log(k+1)$, normalizado. Simula distribuciones de Zipf ligeras.

**C4 — Desbalance severo (geométrico):** $\pi_k^* \propto 2^{-k}$, normalizado. La clase 1 es mucho más grande que las demás.

**C5 — Cotas asimétрicas:** $L_k = 0.5 \cdot \hat{\pi}_k^{\text{natural}}$, $U_k = 1.5 \cdot \hat{\pi}_k^{\text{natural}}$. Restricciones en torno a la distribución natural (siempre factibles).

**C6 — Cotas infactibles:** $L_k = 0.8/K$, $U_k = 1.2/K$ (casi uniforme), aplicado a datasets con distribución muy desbalanceada. Diseñado para que $\hat{\pi} \notin \Pi^*$.

### 9.3 Métrica de Calidad del Clustering Real

Como ground truth para evaluar TCSC, usamos la calidad del clustering con restricciones de tamaño real:

**Algoritmo de clustering:** Size-constrained k-means con método de Zhu et al. (2010) o alternativas como k-means con asignación de transporte óptimo.

**Métricas de evaluación del clustering:**

$$\text{Quality}(D, \Pi^*, \text{alg}) = \text{AMI}(\text{clustering}_{\text{alg}}, y_{\text{true}})$$

donde AMI es Adjusted Mutual Information. También se reportará ARI (Adjusted Rand Index) y el costo de inertia normalizado.

### 9.4 Protocolo Experimental

**Experimento 1 — Correlación de Rango (principal):**

Para cada combinación $(D, \Pi^*_C)$ (5 datasets $\times$ 6 configuraciones = 30 pares):

1. Calcular $\text{TCSC}(D, \Pi^*_C)$ y métricas baseline (CH, SC, DB, CHA, CHA sin $\Psi_{\text{size}}$).
2. Ejecutar el clustering con restricciones (5 semillas, tomar media).
3. Calcular correlación de Spearman entre scores de métricas y $\text{Quality}(D, \Pi^*, \text{alg})$ sobre los 30 pares.

**Hipótesis:** $\rho(\text{TCSC}, \text{Quality}) > \rho(\text{CHA}, \text{Quality}) > \rho(\text{CH}, \text{Quality})$.

**Experimento 2 — Ablación de Componentes:**

Comparar cuatro variantes de TCSC:
- $\text{TCSC}_0$: $\Phi_{\text{struct}}^w = \text{CHA}$, sin $\Psi_{\text{size}}$ (= CHA original)
- $\text{TCSC}_1$: $\Phi_{\text{struct}}^w$ con confianza local, sin $\Psi_{\text{size}}$
- $\text{TCSC}_2$: CHA sin confianza local, con $\Psi_{\text{size}}$
- $\text{TCSC}_3$: completo (propuesta)

Objetivo: demostrar que cada componente contribuye positivamente.

**Experimento 3 — Sensibilidad al Desbalance:**

Fijando un dataset (MNIST), variar la configuración de $\Pi^*$ de C1 a C6. Medir cómo cambia TCSC y correlacionarlo con la degradación real del clustering. Se espera que TCSC capture el efecto de las restricciones infactibles (C6).

**Experimento 4 — Selección de Features:**

Sobre MNIST (features: embeddings de ResNet en 2048D), aplicar el algoritmo greedy (Sección 8.3) para seleccionar subconjuntos de $m \in \{32, 64, 128, 256, 512\}$ features. Comparar contra:
- PCA (sin supervisión, sin considerar $\Pi^*$)
- LDA (supervisado, sin considerar $\Pi^*$)
- Selección por Silhouette (sin restricciones)

Evaluar calidad del clustering restringido en el subespacio seleccionado.

**Experimento 5 — Análisis de Tiempo de Cómputo:**

Medir el tiempo de cómputo de TCSC vs. alternativas (CHA, CH, clasificadores) en función de $n$, $d$ y $K$.

### 9.5 Tabla de Comparación de Métricas

| Métrica | Invariante a dim. | Confianza local | Considera $\Pi^*$ | Rango normalizado |
|---------|--------------------|-----------------|-------------------|-------------------|
| CH | ✗ | ✗ | ✗ | ✗ |
| Silhouette | ✓ (parcial) | ✗ | ✗ | ✓ |
| Davies-Bouldin | ✓ (parcial) | ✗ | ✗ | ✗ |
| CHA (Jeon 2025) | ✓ | ✗ | ✗ | ✓ |
| $\Phi_{\text{struct}}^w$ | ✓ | ✓ | ✗ | ✓ |
| **TCSC (propuesta)** | **✓** | **✓** | **✓** | **✓** |

---

## 10. Ventajas, Limitaciones e Implicaciones

### 10.1 Ventajas

**Teóricas:**
- Sistema axiomático completo (A1–A5) con justificación formal.
- A5 es el primer axioma que incorpora explícitamente la compatibilidad con restricciones de tamaño no uniformes.
- La integración estructural-local (pesos JMDS) está justificada como promedio de Fisher Information local.
- La integración mediante producto respeta las condiciones de necesidad de ambos componentes.

**Prácticas:**
- Permite comparar datasets de distintas dimensiones, cardinalidades y distribuciones — algo que CH, SC y DB no pueden hacer.
- Aplicable directamente a la selección de features para clustering restringido.
- Complejidad lineal en $n$ y $K^2$ (heredada de CHA).
- Flexible ante diferentes formas de $\Pi^*$ (cotas, proporciones, regiones).

**Operacionales:**
- No asume clusters balanceados (a diferencia del enfoque de balanced k-means).
- El único parámetro libre ($k$ en T4-c) puede ser eliminado o auto-calibrado.
- Compatible con cualquier embedding (UMAP, PCA, t-SNE, ResNet features).

### 10.2 Limitaciones

**Estimación del GMM:** En alta dimensionalidad ($d \gg K$), el GMM puede ser inestable. Se recomienda reducción de dimensionalidad previa (PCA a $d' = 2K$ o UMAP a $d' = 2$) o usar GMMs con covarianza diagonal.

**Asignación de pseudo-labels vs. etiquetas:** Si $y_i \neq \hat{y}_i$ (la etiqueta verdadera difiere del pseudo-label del GMM), el cómputo de confianza puede ser inconsistente. Se sugiere usar siempre $y_i$ (la etiqueta verdadera del dataset fuente) y $\hat{y}_i$ solo para GMM.

**Forma de $\Pi^*$:** Si $\Pi^*$ es un conjunto no convexo o una distribución multimodal sobre el simplex, la proyección puede ser computacionalmente costosa o no única.

**Causalidad vs. correlación:** TCSC es una métrica de *correlación* con la calidad del clustering real, no una garantía de causalidad. Un dataset con alto TCSC puede aún producir clustering de baja calidad si el algoritmo de clustering no es adecuado.

**Dependencia de $K$:** Asumir que el número de clusters $K$ es conocido. En la práctica, puede ser necesario un criterio de selección de $K$ previo.

### 10.3 Implicaciones Teóricas

TCSC establece un puente entre tres campos:

1. **Validación de clustering** (CLM, IVM$_A$) — provee la base axiomática para comparar datasets.
2. **Confianza muestral** (JMDS, selección selectiva) — provee la componente local que CLM ignora.
3. **Clustering restringido** (size-constrained k-means) — provee la noción de compatibilidad que los IVM estándar ignoran.

La unión de estos tres enfoques, formalizada en A1–A5, proporciona una teoría más completa para la selección y evaluación de datasets de clasificación en contextos de clustering.

### 10.4 Implicaciones Prácticas

- **Benchmarking de clustering restringido:** TCSC puede usarse para seleccionar y filtrar benchmarks, de forma análoga a como CHA se usa para filtrar benchmarks de clustering libre.
- **Diseño de datasets:** Puede guiar la curación de datasets etiquetados para uso en clustering restringido, identificando qué proporciones son "naturalmente" compatibles con el dataset.
- **Transfer learning para clustering:** En escenarios donde se dispone de múltiples datasets fuente, TCSC puede usarse para ranquear cuál es el más apto para una tarea de clustering restringida en un dominio target.

---

## 11. Resumen Matemático

### 11.1 Notación

| Símbolo | Definición |
|---------|------------|
| $D = \{(x_i, y_i)\}_{i=1}^n$ | Dataset fuente etiquetado |
| $K$ | Número de clusters/clases |
| $\Pi^* \subseteq \Delta^{K-1}$ | Región factible de proporciones |
| $\hat{\pi}_k$ | Proporción natural estimada para cluster $k$ |
| $\gamma_{ik}$ | Responsabilidad GMM de $x_i$ al cluster $k$ |
| $\hat{y}_i$ | Pseudo-label: $\arg\max_k \gamma_{ik}$ |
| $w_i$ | Score de confianza por muestra |
| $\mu_k^w$ | Centroide ponderado por $w_i$ del cluster $k$ |
| $\Phi_{\text{struct}}^w$ | Calidad estructural ponderada por confianza local |
| $\Psi_{\text{size}}$ | Compatibilidad de tamaños |
| TCSC | Métrica de transferibilidad total |

### 11.2 Resumen de Ecuaciones

**Score de confianza estructural:**
$$\text{CS}_{\text{struct}}(x_i) = \frac{\min_{a \neq \hat{y}_i}[\log \gamma_{i\hat{y}_i} - \log \gamma_{ia}]}{\max_j \min_{a \neq \hat{y}_j}[\log \gamma_{j\hat{y}_j} - \log \gamma_{ja}]}$$

**Score de confianza del modelo:**
$$\text{CS}_{\text{model}}(x_i) = p_{g_{-i}}(x_i)_{y_i} \quad (\text{LOOCV})$$

**Peso por muestra:**
$$w_i = \text{CS}_{\text{struct}}(x_i) \cdot \text{CS}_{\text{model}}(x_i)$$

**Centroide ponderado y CH ponderado:**
$$\mu_k^w = \frac{\sum_{i \in C_k} w_i x_i}{\sum_{i \in C_k} w_i}$$

$$\text{CH}_w(S, X, d^2) = \frac{\sum_{k \in S} \left(\sum_{i \in C_k} w_i\right) d^2(\mu_k^w, \mu^w)}{\sum_{k \in S} \sum_{i \in C_k} w_i \cdot d^2(x_i, \mu_k^w)}$$

**Componente estructural:**
$$\Phi_{\text{struct}}^w(D) = \frac{1}{\binom{K}{2}} \sum_{S \subseteq C, |S|=2} \text{CH}_w^5(S, X, d^2)$$

**Distribución natural:**
$$\hat{\pi}_k = \frac{\sum_i \gamma_{ik}}{n}$$

**Proyección a $\Pi^*$:**
$$\pi^* = \arg\min_{\pi \in \Pi^*} \|\hat{\pi} - \pi\|_2^2$$

**Compatibilidad de tamaño:**
$$\Psi_{\text{size}}(D, \Pi^*) = 1 - \frac{\|\hat{\pi} - \pi^*\|_2}{\sqrt{2}}$$

**Métrica TCSC:**
$$\boxed{\text{TCSC}(D, \Pi^*) = \Phi_{\text{struct}}^w(D) \cdot \Psi_{\text{size}}(D, \Pi^*) \in [0, 1]}$$

### 11.3 Pseudocódigo de Implementación

```python
def TCSC(D, labels, Pi_star):
    """
    D: array (n, d) de features
    labels: array (n,) de etiquetas verdaderas
    Pi_star: dict con 'L' y 'U' (cotas), o 'pi_target' (proporción objetivo)
    """
    K = len(set(labels))
    n = len(D)
    
    # --- Paso 1: Ajustar GMM ---
    gmm = GaussianMixture(n_components=K, covariance_type='full')
    gmm.fit(D)
    gamma = gmm.predict_proba(D)  # (n, K)
    pseudo_labels = gamma.argmax(axis=1)
    pi_hat = gamma.mean(axis=0)  # distribución natural
    
    # --- Paso 2: Scores de confianza por muestra ---
    # Confianza estructural (LPG-like)
    log_gamma = np.log(gamma + 1e-10)
    primary = log_gamma[np.arange(n), pseudo_labels]
    secondary = np.partition(-log_gamma, 1, axis=1)[:, 1]
    mingap = primary - (-secondary)
    cs_struct = mingap / mingap.max()
    
    # Confianza del clasificador (MPPL-like, LOOCV con LR)
    lr = LogisticRegressionCV(cv=5)
    lr.fit(D, labels)
    cs_model = lr.predict_proba(D)[np.arange(n), labels]
    
    # Pesos por muestra
    w = cs_struct * cs_model
    
    # --- Paso 3: CHA ponderado ---
    phi = weighted_CHA(D, labels, w, K)
    
    # --- Paso 4: Compatibilidad de tamaño ---
    pi_proj = project_to_feasible(pi_hat, Pi_star, K)
    delta = np.linalg.norm(pi_hat - pi_proj) / np.sqrt(2)
    psi = 1.0 - delta
    
    # --- Paso 5: TCSC ---
    return phi * psi


def project_to_feasible(pi_hat, Pi_star, K):
    """Proyección QP al politopo factible."""
    from scipy.optimize import minimize
    
    L = Pi_star.get('L', np.zeros(K))
    U = Pi_star.get('U', np.ones(K))
    
    res = minimize(
        fun=lambda pi: np.sum((pi - pi_hat)**2),
        x0=pi_hat,
        method='SLSQP',
        bounds=list(zip(L/sum(L), U/sum(U))),
        constraints={'type': 'eq', 'fun': lambda pi: pi.sum() - 1}
    )
    return res.x
```

---

## Referencias Clave

- **Jeon, H. et al. (2025).** "Measuring the Validity of Clustering Validation Datasets." *IEEE TPAMI*, 47(6), 5045–5058.
- **Lee, J. et al. (2022).** "Confidence Score for Source-Free Unsupervised Domain Adaptation." *ICML 2022*, PMLR 162.
- **Ackerman, M. & Ben-David, S. (2008).** "Measures of Clustering Quality: A Working Set of Axioms for Clustering." *NeurIPS 2008*.
- **Duchi, J. et al. (2008).** "Efficient Projections onto the L1-Ball for Learning in High Dimensions." *ICML 2008*.
- **Zhu, S. et al. (2010).** "Constrained Clustering with k-means." *Pattern Recognition*.
- **Abbas, M.M. et al. (2019).** "ClustMe: A Visual Quality Measure for Ranking Monochrome Scatterplots Based on Cluster Patterns." *CGF*, 38(3).

---

*Documento preparado para tesis: "Diseño de una Métrica de Transferencia para la Utilización de Conjuntos de Datos de Clasificación en Tareas de Clustering" — versión de trabajo.*