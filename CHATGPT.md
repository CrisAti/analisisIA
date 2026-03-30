# Propuesta de Métricas de Transferencia para usar Datasets Etiquetados de Clasificación en Clustering con Restricciones Generales de Tamaño

## Tesis
**“Diseño de una Métrica de Transferencia para la Utilización de Conjuntos de Datos de Clasificación en Tareas de Clustering”.**

---

## 1. Problema formal

Sea un dataset etiquetado fuente

```math
\mathcal D_s=\{(x_i,y_i)\}_{i=1}^n,\qquad y_i\in\{1,\dots,K\}.
```

Queremos estimar, **antes** de ejecutar clustering restringido, qué tan apto es este dataset para transferirse a una tarea de clustering con restricciones generales de tamaño. La salida deseada es una métrica

```math
\mathcal T(\mathcal D_s,\Pi^{*})
\in [0,1]
```

tal que valores altos indiquen que:

1. las etiquetas del dataset capturan una estructura agrupable real;
2. esa estructura es confiable tanto **globalmente** como **localmente**;
3. dicha estructura es **compatible** con una región factible de tamaños $\Pi^{*}$, que puede tomar formas como:
   - cotas por cluster: $n_k \in [L_k, U_k]$,
   - proporciones objetivo no uniformes: $`\pi^{*} = (\pi_1^{*}, \dots, \pi_K^{*}), \qquad \sum_k \pi_k^{*} = 1`$,
   - una región factible general: $\Pi^{*} \subset \Delta^{K-1}$.

Aquí $\Delta^{K-1}$ es el simplex de probabilidad.

---

## 2. Idea central: una métrica de transferencia debe unir tres niveles

La propuesta no debe tratar CLM y JMDS como bloques pegados artificialmente. La forma correcta de integrarlos es pensar que la transferibilidad tiene tres niveles acoplados:

### Nivel 1: estructura global
Las etiquetas solo son útiles para clustering si inducen particiones compactas y separadas de forma estable **a través de datasets**, no solo dentro de un dataset particular.

### Nivel 2: confiabilidad local
No todas las muestras sostienen con la misma fuerza esa estructura. Algunas están en núcleos de su clase; otras en regiones ambiguas, bordes o solapamientos. La métrica debe ponderar el soporte estructural muestra a muestra.

### Nivel 3: compatibilidad con restricciones de tamaño
Incluso si las etiquetas reflejan bien la estructura, esa estructura puede ser mala candidata para clustering restringido si sus masas naturales son incompatibles con la región factible de tamaños objetivo.

De aquí se desprende una familia de métricas de transferencia:

```math
\mathcal T = \mathrm{Fuse}\big(\text{estructura global ajustada},\ \text{confiabilidad local integrada},\ \text{compatibilidad de tamaño}\big).
```

La palabra clave es **integrada**: la confiabilidad local no se añade al final, sino que modifica desde el inicio la forma en que medimos estructura y masas naturales.

---

## 3. Qué tomamos de CLM y de JMDS

## 3.1. De CLM

Del marco CLM se debe conservar el espíritu axiomático, no solo un índice puntual. Los cuatro principios base son:

- invariancia a cardinalidad del dataset;
- invariancia a desplazamiento de distancias;
- invariancia al número de clases;
- invariancia de rango.

Esto es esencial porque una métrica de transferencia no debe subir o bajar artificialmente por tener más muestras, más dimensiones o más clases, sino por la calidad estructural del dataset.

## 3.2. De JMDS

La idea valiosa de JMDS no es únicamente la fórmula $\mathrm{LPG}\cdot \mathrm{MPPL}$, sino que la confianza debe construirse combinando:

- **conocimiento estructural del dato**;
- **conocimiento del modelo/etiqueta**;
- y debe ser **local por muestra**.

En esta tesis, esa intuición se reutiliza así: cada muestra recibe una confiabilidad $r_i$ que combina evidencia estructural y evidencia supervisada, y luego $r_i$ afecta tanto el término global como el término de compatibilidad de tamaño.

---

## 4. Extensión axiomática: nuevo principio para restricciones de tamaño

Los axiomas A1–A4 no cubren la relación entre la estructura natural del dataset y una región factible de tamaños. Por eso se propone un nuevo axioma.

## A5. Invariancia / Monotonicidad por Compatibilidad de Tamaño

Sea $m(C)\in \Delta^{K-1}$ el vector de masas naturales inducido por una partición $C$, con

```math
m_k(C)=\frac{|C_k|}{\sum_{j=1}^K |C_j|}.
```

Sea $d_\Pi(m,\Pi^{*})$ una distancia de proyección desde $m$ hacia la región factible $\Pi^{*}$.

Una métrica de transferencia $\mathcal T(C,X,\delta;\Pi^{*})$ satisface el axioma A5 si:

### A5.1. Máximo en factibilidad exacta
```math
m(C)\in \Pi^{*} \quad \Longrightarrow \quad \mathcal T_{\text{size}}(C;\Pi^{*})=1.
```

### A5.2. Monotonicidad respecto a la distancia factible
Si
```math
d_\Pi(m(C),\Pi_1^{*}) \le d_\Pi(m(C),\Pi_2^{*}),
```
entonces
```math
\mathcal T_{\text{size}}(C;\Pi_1^{*}) \ge \mathcal T_{\text{size}}(C;\Pi_2^{*}).
```

### A5.3. Invariancia a la parametrización de la región factible
Si dos descripciones $\Pi_a^{*}$ y $\Pi_b^{*}$ representan exactamente la misma región factible en el simplex, entonces:
```math
\mathcal T_{\text{size}}(C;\Pi_a^{*})=\mathcal T_{\text{size}}(C;\Pi_b^{*}).
```

### Por qué A5 no está cubierto por A1–A4

- A1 trata tamaño del dataset, no compatibilidad entre **masas naturales** y **restricciones externas**.
- A2 trata desplazamiento de distancias, no distribución de tamaños.
- A3 trata cantidad de clases, no forma de la región factible.
- A4 trata comparabilidad del rango, no factibilidad estructural.

Es decir: un dataset puede satisfacer perfectamente A1–A4 y aun así ser pésimo para clustering restringido si la estructura natural contradice las restricciones de tamaño.

---

## 5. Construcción de una confiabilidad local integrada

Sea $z_i=\phi(x_i)\in\mathbb R^d$ una representación del dato. Puede ser:

- el espacio original;
- un embedding auto-supervisado;
- o, preferiblemente, una representación aprendida por clasificación.

Queremos una confiabilidad local $r_i\in[0,1]$ que combine evidencia:

1. **supervisada/modelo**;
2. **estructural/local**.

## 5.1. Evidencia supervisada

Entrenamos un clasificador $p_M(y\mid z)$. Para la etiqueta observada $y_i$:

```math
u_i = p_M(y_i\mid z_i).
```

Este término mide cuánto respalda el modelo discriminativo a la etiqueta de la muestra.

## 5.2. Evidencia estructural

Construimos una distribución estructural $p_S(y\mid z_i)$ usando, por ejemplo:

- GMM por clase en el espacio $z$;
- k-NN label density;
- o mezcla local basada en grafo.

Una opción simple y robusta es:

```math
v_i = p_S(y_i\mid z_i).
```

También se puede reforzar con un margen estructural:

```math
g_i = \frac{\ell^{(1)}_i-\ell^{(2)}_i}{|\ell^{(1)}_i|+\varepsilon},
```

donde $\ell^{(1)}_i$ y $\ell^{(2)}_i$ son el mayor y segundo mayor log-posterior estructural. Entonces se define:

```math
v_i^\text{margin} = \frac{1+g_i}{2}\in[0,1].
```

## 5.3. Fusión local recomendada

No conviene usar simplemente un producto. La mejor opción práctica es una **media armónica**, porque exige que ambas evidencias sean altas, pero sin la fragilidad extrema del producto:

```math
r_i = \frac{2u_i v_i}{u_i+v_i+\varepsilon}.
```

Si se incluye el margen:

```math
r_i = \frac{3}{\frac{1}{u_i+\varepsilon}+\frac{1}{v_i+\varepsilon}+\frac{1}{v_i^\text{margin}+\varepsilon}}.
```

### Interpretación
- Si una muestra es “fácil” para clasificación pero estructuralmente ambigua, su peso baja.
- Si una muestra cae en un núcleo geométrico pero el modelo no respalda la etiqueta, su peso también baja.
- Los núcleos consistentes de clase obtienen mayor $r_i$.

Esto hereda la intuición de JMDS: una muestra confiable debe serlo simultáneamente desde el punto de vista del modelo y de la estructura.

---

## 6. Propuesta principal: métrica RTM-H (Reliability-aware Transfer Metric, versión armónica)

Esta es la propuesta más recomendable para una tesis porque es interpretable, implementable y alineada con CLM + JMDS + restricciones de tamaño.

## 6.1. Distancias ponderadas por confiabilidad

Sea $d_{ij}=\|z_i-z_j\|^2$. Para cada par de clases $(a,b)$, definimos promedios ponderados:

```math
\bar d_{aa}^{(r)}=
\frac{\sum_{i:y_i=a}\sum_{j:y_j=a,\ j\neq i} r_i r_j d_{ij}}
{\sum_{i:y_i=a}\sum_{j:y_j=a,\ j\neq i} r_i r_j},
```

```math
\bar d_{bb}^{(r)}=
\frac{\sum_{i:y_i=b}\sum_{j:y_j=b,\ j\neq i} r_i r_j d_{ij}}
{\sum_{i:y_i=b}\sum_{j:y_j=b,\ j\neq i} r_i r_j},
```

```math
\bar d_{ab}^{(r)}=
\frac{\sum_{i:y_i=a}\sum_{j:y_j=b} r_i r_j d_{ij}}
{\sum_{i:y_i=a}\sum_{j:y_j=b} r_i r_j}.
```

Esto hace que el score global se apoye más en las regiones localmente confiables.

## 6.2. Score estructural local por par de clases

Inspirados por el protocolo de invariancia por desplazamiento, definimos:

```math
\psi_{ab} =
\frac{
\exp\left(\bar d_{ab}^{(r)}/s_{ab}\right)
}{
\exp\left(\bar d_{ab}^{(r)}/s_{ab}\right) +
\exp\left(\frac{\bar d_{aa}^{(r)}+\bar d_{bb}^{(r)}}{2s_{ab}}\right)
},
```

donde $s_{ab}$ es una escala auto-calibrada, por ejemplo la desviación estándar de las distancias entre muestras de las clases $a$ y $b$.

### Lectura de $\psi_{ab}$
- sube cuando la distancia entre clases supera la dispersión intra-clase;
- cae si las clases están mezcladas o internamente muy dispersas.

## 6.3. Score estructural global

Agregamos por pares de clases: $G_r = \frac{2}{K(K-1)} \sum_{a<b} \psi_{ab}$.`

Esto mantiene el espíritu de A3: el score global depende de la calidad media por pares, no del mero conteo de clases.

## 6.4. Ajuste de rango (estilo CLM)

Para no comparar scores crudos entre datasets, usamos un baseline aleatorio que preserve las proporciones de clase. Sea $\Pi_\text{perm}$ el conjunto de particiones aleatorias que preservan cardinalidades de clase, y definimos:

```math
G_{\text{rand}} = \mathbb E_{\pi\sim \Pi_\text{perm}}[G_r(\pi)].
```

Entonces el score estructural ajustado es

```math
S_{\text{struct}} =
\left[
\frac{G_r - G_{\text{rand}}}{1-G_{\text{rand}}+\varepsilon}
\right]_{[0,1]},
```

donde $[x]_{[0,1]}=\min(1,\max(0,x))$.

### Comentario
Esto es coherente con CLM:
- A1: usa estimadores promedio robustos a subsampling proporcional;
- A2: el ratio exponencial cancela desplazamientos aditivos;
- A3: agrega por pares de clases;
- A4: reescala a $[0,1]$ usando un baseline comparable.

---

## 7. Compatibilidad general con restricciones de tamaño

La parte nueva de la tesis es que el dataset no debe ser solo “clusterizable”, sino también **transferible a clustering restringido**.

## 7.1. Masas naturales confiables

No usamos las cardinalidades brutas de clase. Usamos masas efectivas ponderadas por confiabilidad:

```math
m_k^{(r)} =
\frac{\sum_{i=1}^n r_i \mathbf 1[y_i=k]}
{\sum_{i=1}^n r_i}.
```

Definimos

```math
m^{(r)}=(m_1^{(r)},\dots,m_K^{(r)})\in\Delta^{K-1}.
```

Esto tiene una interpretación fuerte: no todas las muestras aportan igual evidencia sobre el “tamaño natural” de un cluster potencial.

## 7.2. Región factible general

Los requerimientos pueden escribirse como:

```math
\Pi^{*}=\{q\in \Delta^{K-1}:\ Aq\le b,\ Cq=d\},
```

lo que incluye:

- bounds inferiores/superiores;
- proporciones objetivo;
- mezclas de restricciones lineales.

## 7.3. Distancia a la región factible

### Opción recomendada: proyección por Jensen-Shannon
Para restricciones distribucionales:

```math
q^{*} = \arg\min_{q\in \Pi^{*}} \mathrm{JS}(m^{(r)}\|q),
```

```math
D_{\text{size}} = \mathrm{JS}(m^{(r)}\|q^{*}).
```

Si se usa log base 2, entonces:

```math
0\le \mathrm{JS}(\cdot,\cdot)\le 1.
```

Así, el score de tamaño queda:

```math
S_{\text{size}} = 1 - D_{\text{size}}.
```

### Opción alternativa: proyección euclídea
Si $\Pi^{*}$ es un politopo simple:

```math
D_{\text{size}}^{(2)} = \min_{q\in\Pi^{*}}\|m^{(r)}-q\|_2,
```

```math
S_{\text{size}}^{(2)} =
1-\frac{D_{\text{size}}^{(2)}}{D_{\max}(\Pi^{*})}.
```

### Opción alternativa: costo de transporte
Si se quiere capturar “cuánta masa habría que mover”:

```math
D_{\text{size}}^{\text{OT}}
=
\min_{q\in\Pi^{*}}\mathrm{OT}(m^{(r)},q).
```

Esto es especialmente útil cuando las clases tienen semejanzas semánticas y no todo movimiento de masa debería costar igual.

---

## 8. Fusión final recomendada

La métrica final debe exigir simultáneamente:

- buena estructura;
- buena compatibilidad de tamaño.

La fusión recomendada es una **media armónica**:

```math
\boxed{
\mathcal T_{\text{RTM-H}}
=
\frac{2\,S_{\text{struct}}\,S_{\text{size}}}
{S_{\text{struct}}+S_{\text{size}}+\varepsilon}
}
```

### Justificación
No es una multiplicación arbitraria:
- la media armónica modela un requisito de tipo “conjunción”;
- penaliza que uno de los dos componentes sea muy bajo;
- pero es menos frágil que el producto;
- no requiere hiperparámetros externos.

### Interpretación
- Si el dataset tiene muy buena CLM pero masas incompatibles con $\Pi^{*}$, la transferibilidad baja.
- Si las masas encajan perfecto pero la estructura está mal alineada con las etiquetas, también baja.
- Solo sube cuando ambas condiciones se cumplen.

---

## 9. Dos caminos alternativos adicionales

---

## 9.1. RTM-OT: métrica por transporte óptimo confiable

Esta variante integra localmente estructura, etiquetas y restricciones de tamaño en un mismo problema.

### Paso 1: costo muestra-cluster
Sea $k$ un cluster objetivo. Definimos:

```math
c_{ik}
=
-\log p_M(k\mid z_i)
-\log p_S(k\mid z_i).
```

También puede escribirse con ponderación armónica o suma de logits, pero sin calibración externa la suma de costos negativos logarítmicos es muy natural.

### Paso 2: plan de asignación restringido
Buscamos una matriz $P\in\mathbb R_+^{n\times K}$:

```math
\min_{P,q\in\Pi^{*}}
\sum_{i=1}^n \sum_{k=1}^K P_{ik} c_{ik}
+
\tau \sum_{i,k} P_{ik}\log P_{ik}
```

sujeto a:

```math
\sum_{k=1}^K P_{ik}=\frac1n,\qquad
\sum_{i=1}^n P_{ik}=q_k.
```

### Score
```math
\mathcal T_{\text{RTM-OT}} = 1-\frac{\mathcal C^{*}-\mathcal C_{\min}}{\mathcal C_{\max}-\mathcal C_{\min}+\varepsilon}.
```

### Ventajas
- integra tamaño desde el núcleo del modelo;
- da asignaciones suaves;
- produce explicaciones a nivel muestra-cluster.

### Limitaciones
- costo computacional mayor;
- requiere resolver OT/Sinkhorn;
- más difícil de explicar si la tesis busca una métrica sencilla.

---

## 9.2. RTM-Graph: métrica basada en grafo y cortes restringidos

### Construcción
Construimos un grafo de afinidad $W$ sobre los embeddings $z_i$. Para la partición dada por las etiquetas, definimos un normalized cut confiable:

```math
\mathrm{NCut}_r(C)=
\sum_{k=1}^K
\frac{\mathrm{cut}_r(C_k,\bar C_k)}{\mathrm{vol}_r(C_k)},
```

donde

```math
\mathrm{cut}_r(A,B)=\sum_{i\in A, j\in B} r_i r_j W_{ij},
\qquad
\mathrm{vol}_r(A)=\sum_{i\in A,j} r_i r_j W_{ij}.
```

Luego se combina con una penalización de factibilidad:

```math
\mathcal T_{\text{RTM-Graph}}
=
1-\alpha \,\widetilde{\mathrm{NCut}}_r
-(1-\alpha)\,D_{\text{size}}.
```

### Recomendación
Útil como línea secundaria, especialmente si la tesis explora subespacios o selección de features mediante grafos. Pero no es mi primera opción como propuesta central porque introduce un hiperparámetro $\alpha$.

---

## 10. Propiedades teóricas de RTM-H

## 10.1. Respecto a A1–A4

### A1. Invariancia a cardinalidad
Los términos $\bar d_{aa}^{(r)}, \bar d_{ab}^{(r)}$ son medias ponderadas. Bajo subsampling proporcional por clase, convergen al mismo valor poblacional.

### A2. Invariancia a desplazamiento
Si todas las distancias se desplazan por una constante $\beta$, el cociente exponencial de $\psi_{ab}$ conserva el orden relativo entre separación inter-clase y dispersión intra-clase.

### A3. Invariancia al número de clases
El score global promedia sobre pares de clases.

### A4. Invariancia de rango
El ajuste con $G_{\text{rand}}$ y el clipping a $[0,1]$ hacen el score comparable entre datasets.

## 10.2. Respecto a A5

```math
m^{(r)}\in\Pi^{*} \Rightarrow S_{\text{size}}=1
\Rightarrow
\mathcal T_{\text{RTM-H}}
=
\frac{2S_{\text{struct}}}{1+S_{\text{struct}}}
```

y si además $S_{\text{struct}}=1$, entonces $\mathcal T_{\text{RTM-H}}=1$.

Si la distancia a $\Pi^{*}$ crece, $S_{\text{size}}$ decrece monótonamente y por ende también lo hace $\mathcal T_{\text{RTM-H}}$.

---

## 11. Diseño experimental para validar la métrica

## 11.1. Datasets

Usar:

- MNIST
- Fashion-MNIST
- CIFAR-10
- CIFAR-100 (opcional, con agrupación por superclases)
- STL-10 o Tiny-ImageNet como extensión

## 11.2. Espacios de representación

Para no sesgar el resultado a un único espacio, probar al menos tres representaciones:

1. píxeles reducidos por PCA;
2. embeddings auto-supervisados;
3. embeddings de una red entrenada para clasificación.

La métrica debe evaluarse sobre el espacio donde se planea ejecutar clustering.

## 11.3. Generación de restricciones de tamaño

No centrarse en balanced k-means. Diseñar varios regímenes:

### Régimen A: cotas amplias
```math
L_k = 0.6\,n\pi_k,\qquad U_k = 1.4\,n\pi_k.
```

### Régimen B: proporciones no uniformes
Elegir objetivos como:

```math
\pi^{*}=(0.40,0.25,0.15,0.10,0.10)
```

para $K=5$, o análogos para otros $K$.

### Régimen C: región factible
Definir poliedros con:
- bounds por grupo,
- suma parcial de ciertos clusters,
- o restricciones del tipo:
  ```math
  \pi_1+\pi_2\le 0.45.
  ```

### Régimen D: factibilidad degradada
Tomar la masa natural estimada $m^{(r)}$ y alejarla gradualmente hacia regiones menos compatibles mediante interpolación:

```math
\pi^{*}(\lambda)= (1-\lambda)m^{(r)}+\lambda \tilde \pi,
\qquad \lambda\in[0,1].
```

Esto permite sensibilidad controlada.

## 11.4. Algoritmos de clustering restringido

Comparar con varios algoritmos, no solo uno:

- constrained k-means con bounds;
- COP-k-means si se desea incorporar restricciones adicionales;
- balanced / capacity-constrained variants solo como caso particular;
- métodos de asignación por min-cost flow;
- variantes con Sinkhorn o OT si se implementa.

## 11.5. Definición de “calidad real” del clustering restringido

Para cada dataset y conjunto de restricciones, correr clustering restringido y medir:

- NMI / AMI frente a las etiquetas;
- ARI;
- pureza;
- costo intra-cluster;
- violación de restricciones (debe ser 0 o muy pequeña);
- estabilidad entre reinicios.

Definir un score compuesto real:

```math
Q_{\text{real}} = \eta_1 \,\mathrm{AMI}
+\eta_2 \,\mathrm{ARI}
+\eta_3 \,(1-\widetilde{\text{cost}})
+\eta_4 \,\mathrm{stability},
```

o alternativamente trabajar con cada métrica por separado para evitar arbitrariedad.

## 11.6. Validación principal

Para cada dataset $D$, representación $\phi$, y restricción $\Pi^{*}$:

1. calcular $\mathcal T_{\text{RTM-H}}(D,\Pi^{*})$;
2. ejecutar clustering restringido;
3. medir calidad real $Q_{\text{real}}$.

Luego medir:

- correlación de Spearman entre $\mathcal T$ y $Q_{\text{real}}$;
- correlación de Kendall;
- calibración ordinal top-k;
- error al ordenar datasets por transferibilidad real.

## 11.7. Baselines

Comparar contra:

### Baselines globales clásicos
- Calinski-Harabasz
- Silhouette
- Davies-Bouldin
- Dunn
- índices ajustados tipo CLM si los implementas

### Baselines locales / híbridos
- promedio de márgenes del clasificador;
- entropía media;
- score tipo JMDS adaptado de forma naive;
- pura compatibilidad de tamaño $S_{\text{size}}$ sin estructura;
- pura estructura $S_{\text{struct}}$ sin tamaño.

Lo importante es mostrar que:
1. un score global sin tamaño es insuficiente;
2. un score de tamaño sin estructura es insuficiente;
3. una mezcla naive también es peor que la integración propuesta.

## 11.8. Análisis de sensibilidad

Hacer curvas respecto a:

- intensidad del desbalance;
- amplitud de bounds $[L_k,U_k]$;
- distancia entre masa natural y masa objetivo;
- calidad del embedding;
- cantidad de clases $K$;
- presencia de clases multimodales.

## 11.9. Ablación

Probar:

1. sin confiabilidad local: $r_i=1$;
2. solo evidencia supervisada: $r_i=u_i$;
3. solo evidencia estructural: $r_i=v_i$;
4. fusión por producto;
5. fusión por media armónica;
6. masa bruta $m$ vs masa confiable $m^{(r)}$.

Esto es crítico para demostrar que la integración local realmente aporta.

---

## 12. Uso para selección de subespacios o features

Aquí la métrica puede convertirse en criterio de búsqueda de subespacios:

```math
S^{*} = \arg\max_{S\subset\{1,\dots,p\}} \mathcal T(\mathcal D_s^S,\Pi^{*}),
```

donde $\mathcal D_s^S$ es el dataset restringido a las features $S$.

## 12.1. Procedimiento

1. seleccionar un subconjunto de features;
2. calcular embeddings o trabajar en el subespacio directamente;
3. computar $r_i$, $S_{\text{struct}}$, $S_{\text{size}}$;
4. evaluar $\mathcal T$;
5. usar búsqueda greedy, forward selection, beam search o algoritmo genético.

## 12.2. Interpretación

Un subespacio es bueno si simultáneamente:

- hace las etiquetas más coherentes con la geometría;
- reduce regiones ambiguas;
- y produce masas naturales más compatibles con la restricción.

Esto es superior a seleccionar features solo con clasificación o solo con clustering clásico.

## 12.3. Variante regularizada

Si se quiere evitar subespacios grandes:

```math
\mathcal T_{\lambda}(S)=\mathcal T(S)-\lambda\frac{|S|}{p}.
```

Si no se desean hiperparámetros libres, usar selección por presupuesto fijo de features en lugar de penalización.

---

## 13. Ventajas y limitaciones

## Ventajas

- integra CLM y confianza local en una sola construcción;
- incorpora restricciones de tamaño no uniformes;
- no depende de balanced clustering;
- usa componentes interpretables;
- evita calibración externa fuerte;
- puede emplearse para ranking de datasets, embeddings o subespacios.

## Limitaciones

- depende de la calidad del embedding $\phi$;
- si las etiquetas tienen semántica muy distinta de la geometría visual, el score puede ser bajo incluso con alta utilidad para clasificación;
- el cálculo de $G_{\text{rand}}$ requiere permutaciones Monte Carlo;
- para datasets muy grandes, las distancias por pares requieren aproximaciones.

## Soluciones prácticas
- usar mini-batches o muestreo por clase para estimar distancias;
- usar k-NN graph en vez de todas las parejas;
- estimar $G_{\text{rand}}$ con 20–50 permutaciones;
- usar PCA previa o embeddings compactos.

---

## 14. Recomendación final para la tesis

Si tuviera que elegir una propuesta central para desarrollar, implementaría:

```math
\boxed{\mathcal T_{\text{RTM-H}}}
```

porque cumple mejor con lo que necesitas:

1. mantiene el espíritu axiomático de CLM;
2. integra confianza local al estilo JMDS, pero sin separarla como bloque externo;
3. incorpora formalmente restricciones generales de tamaño mediante A5;
4. es suficientemente concreta para tesis, experimentación y ablation.

Usaría además:

- **RTM-OT** como extensión teórica fuerte;
- **RTM-Graph** como línea complementaria para subspace selection.

---

## 15. Forma resumida de la propuesta principal

### Paso 1
Obtener embeddings $z_i=\phi(x_i)$.

### Paso 2
Construir evidencia supervisada $u_i$ y estructural $v_i$.

### Paso 3
Fusionarlas en una confiabilidad local:

```math
r_i = \frac{2u_i v_i}{u_i+v_i+\varepsilon}.
```

### Paso 4
Calcular el score estructural ajustado: $S_{\text{struct}} = \Big[\frac{\frac{2}{K(K-1)}\sum_{a<b}\psi_{ab} - G_{\text{rand}}}{1-G_{\text{rand}}+\varepsilon}\Big]_{[0,1]}$.

### Paso 5
Calcular masa confiable:

```math
m_k^{(r)} =
\frac{\sum_i r_i \mathbf 1[y_i=k]}
{\sum_i r_i}.
```

### Paso 6
Proyectar $m^{(r)}$ hacia $\Pi^{*}$ y obtener:

```math
S_{\text{size}} = 1-\min_{q\in\Pi^{*}}\mathrm{JS}(m^{(r)}\|q).
```

### Paso 7
Fusionar:

```math
\boxed{
\mathcal T_{\text{RTM-H}}
=
\frac{2S_{\text{struct}}S_{\text{size}}}
{S_{\text{struct}}+S_{\text{size}}+\varepsilon}
}
```

---

## 16. Hipótesis de investigación derivables

1. **H1:** $\mathcal T_{\text{RTM-H}}$ correlaciona mejor con la calidad real del clustering restringido que CH, Silhouette y Davies-Bouldin.
2. **H2:** incorporar confiabilidad local mejora la correlación frente a una versión puramente global.
3. **H3:** usar masa confiable $m^{(r)}$ supera a usar proporciones brutas de clase.
4. **H4:** la compatibilidad con $\Pi^{*}$ explica parte de la varianza que las métricas globales tradicionales no capturan.
5. **H5:** maximizar $\mathcal T_{\text{RTM-H}}$ sobre subespacios produce representaciones más útiles para clustering restringido.

---

## 17. Cierre

La idea más fuerte para tu tesis es reinterpretar la “transferibilidad” no como una simple semejanza entre clasificación y clustering, sino como una **compatibilidad triple**:

- compatibilidad entre etiquetas y estructura global;
- compatibilidad entre etiquetas y soporte local por muestra;
- compatibilidad entre la masa natural inducida por esa estructura y la región factible de tamaños.

Esa es precisamente la parte novedosa: pasar de “¿las clases parecen clusters?” a

```math
\text{“¿las clases inducen una estructura confiable y utilizable bajo restricciones generales de tamaño?”}
```

y medirlo formalmente con una métrica implementable.
