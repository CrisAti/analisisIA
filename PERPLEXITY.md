# Métrica de transferencia desde clasificación hacia clustering con restricciones de tamaño basada en CLM y JMDS

## 1. Planteamiento y contexto

En muchos trabajos de clustering se utilizan datasets etiquetados como "ground truth" para validar algoritmos no supervisados, asumiendo que las clases coinciden razonablemente con la estructura de clusters latentes.
Esta suposición se formaliza como el supuesto de *Cluster-Label Matching (CLM)*.[^1][^2][^3][^4]

Sin embargo, en tareas reales de *size-constrained clustering* (clustering con restricciones de tamaño), el objetivo no es solo recuperar las clases, sino obtener una partición que satisfaga restricciones de tamaño potencialmente no uniformes sobre los clusters, como cotas inferiores/superiores o vectores de proporciones objetivo arbitrarios.[^5][^6][^7]
En este contexto, la utilidad de un dataset etiquetado como fuente para tareas de clustering (transferencia clasificación→clustering) depende de tres aspectos acoplados:

- Qué tan bien las etiquetas reflejan la estructura intrínseca de clusters (CLM global).
- Qué tan confiable es la estructura local alrededor de cada muestra (idea de confianza local tipo JMDS).
- Qué tan compatible es la distribución de tamaños inducida por las etiquetas con las restricciones de tamaño objetivo.

El objetivo de este documento es proponer un marco teórico y práctico para una métrica de transferencia
\(T(D,\Pi^*)\) que cuantifique la aptitud de un dataset etiquetado \(D\) para ser utilizado en tareas de clustering con restricciones de tamaño especificadas por una región factible \(\Pi^*\) en el simplex de tamaños.
La propuesta integra:

- El marco axiomático de CLM (axiomas A1–A4) a través de índices internos ajustados (IVMAs).
- La noción de confianza local/muestral introducida por el *Joint Model-Data Structure (JMDS) score* en adaptación de dominio sin fuente.[^8][^9]
- Un nuevo axioma de compatibilidad de tamaño (A5) que extiende CLM para incorporar restricciones de tamaño generales, no necesariamente uniformes.

## 2. Fundamentos: CLM e IVMAs

### 2.1. Axiomas de CLM

Jeon et al. proponen medir la validez de datasets de validación de clustering a través del nivel de *cluster-label matching (CLM)*, introduciendo cuatro axiomas que un índice interno ajustado debe satisfacer para ser comparable entre datasets.[^2][^3][^4][^1]
Dados un dataset etiquetado \(D = \{(x_i,y_i)\}_{i=1}^N\), un índice interno \(V\) y su versión ajustada \(V^{\text{adj}}\), los axiomas son:

- **A1 – Invariancia a cardinalidad de datos (Data-Cardinality Invariance).**  La medida no debe variar de forma significativa con el tamaño del dataset; idealmente, duplicar o muestrear aleatoriamente el dataset no debería cambiar el valor de la métrica.[^3][^2]
- **A2 – Invariancia a desplazamiento (Shift Invariance).**  La medida debe ser robusta a cambios en la escala o desplazamientos en el espacio de distancias entre datasets, típicamente mediante transformaciones exponenciales o logísticas sobre el índice base.[^4][^2]
- **A3 – Invariancia al número de clases (Class-Cardinality Invariance).**  El valor de la métrica no debe depender directamente del número de clases, de modo que dos datasets bien clusterizados con distinto número de clases obtengan valores comparables.[^2]
- **A4 – Invariancia de rango (Range Invariance).**  La escala de la métrica debe estar normalizada (por ejemplo, a \([0,1]\)) para que los valores puedan compararse entre datasets heterogéneos.[^3][^2]

Para obtener índices que satisfacen estos axiomas, Jeon et al. proponen protocolos de ajuste (T1–T4) que aplican estimadores robustos, transformaciones exponenciales, agregación a nivel de ejemplo/clase y normalización de rango para convertir índices internos clásicos (Silhouette, Calinski–Harabasz, etc.) en *Adjusted Internal Validation Measures* (IVMAs).[^2][^3]

### 2.2. Índice CLM global derivado de IVMAs

Sea \(V\) un índice interno estándar (por ejemplo, Calinski–Harabasz) calculado sobre la partición inducida por las etiquetas \(y\).
Aplicando los protocolos T1–T4, se obtiene una versión ajustada \(V^{\text{adj}}\) que satisface A1–A4 y está normalizada a \([0,1]\).[^2]
Definimos entonces:

\[
Q_{\text{CLM}}(D) \in [0,1]
\]

como el valor de un IVMA escogido (por ejemplo, CH ajustado) calculado sobre la partición verdadera \(\{y_i\}\).
Este término representa la calidad global de CLM del dataset, independientemente de restricciones de tamaño.

## 3. Fundamentos: JMDS y confianza local

### 3.1. JMDS en adaptación de dominio sin fuente

En *source-free unsupervised domain adaptation* (SFUDA), Lee et al. introducen el *Joint Model-Data Structure (JMDS) score* para cuantificar la confianza por muestra combinando información del modelo y de la estructura de datos.[^9][^8]
La idea es:

- Ajustar un modelo de mezcla gaussiana (GMM) en el espacio de características del dominio objetivo para obtener, para cada muestra, las probabilidades de pertenecer a cada componente (clase pseudo-etiquetada).
- Definir el **Log-Probability Gap (LPG)** como la diferencia entre los log-likelihoods del componente más probable y el segundo más probable para cada muestra.[^8]
- Definir el **Model Probability of Pseudo-Label (MPPL)** como la probabilidad que el modelo de clasificación asigna a la pseudo-etiqueta elegida para la muestra.[^8]
- Definir el **JMDS score** por muestra como el producto normalizado de estos dos términos para enfatizar aquellas muestras que son simultáneamente consistentes con la estructura de datos (LPG alto) y con el modelo (MPPL alto).[^9][^8]

Este enfoque introduce explícitamente la noción de confianza local o por muestra, y sugiere ponderar las regiones del espacio de características según dicha confianza, en lugar de tratar el dataset como homogéneamente fiable.

### 3.2. Adaptación del espíritu JMDS a la métrica propuesta

Aunque JMDS está formulado para SFUDA, la idea clave de combinar:

- una medida local de compatibilidad con la estructura no supervisada (GMM / clustering en el espacio de características), y
- una medida local de confianza del modelo supervisado entrenado con las etiquetas,

es directamente aplicable al problema de medir la transferibilidad de un dataset etiquetado hacia clustering.
En lugar de usar JMDS como bloque separado, se propone incorporarlo en la definición de un factor de confianza local \(\ell_i\) que modula la contribución de cada ejemplo a la métrica de transferencia.

## 4. Clustering con restricciones de tamaño

### 4.1. Formulación del problema con \(\Pi^*\)

Considérese un problema de clustering con \(K\) clusters sobre un conjunto de \(N\) ejemplos, con restricciones de tamaño expresadas como una región factible \(\Pi^*\) en el simplex de proporciones de tamaño:

\[
\Delta^{K-1} = \left\{ \pi \in \mathbb{R}^K_{\ge 0} : \sum_{k=1}^K \pi_k = 1 \right\}.
\]

Ejemplos de \(\Pi^*\):

- **Cotas inferiores/superiores por cluster** \(L_k, U_k\):
  \[
  \Pi^* = \left\{ \pi \in \Delta^{K-1} : \tfrac{L_k}{N} \le \pi_k \le \tfrac{U_k}{N} \; \forall k \right\}.
  \]
- **Proporciones objetivo no uniformes** \(\pi^{*} \in \Delta^{K-1}\): por ejemplo, \(\pi^{*} = (0.05,0.10,\dots)\), que puede verse como una región puntual \(\Pi^* = \{\pi^{*}\}\) o como una vecindad alrededor de \(\pi^{*}\).
- **Regiones factibles generales** definidas por restricciones lineales adicionales (por ejemplo, grupos de clusters que comparten capacidad).

Los algoritmos de clustering con restricciones de tamaño (por ejemplo, variantes de k-means con cotas \(L_k, U_k\), enfoques basados en *minimum-cost flow* o modelos probabilísticos de micro-clustering con tamaños máximos) tratan típicamente este problema como uno de optimización combinatoria sobre asignaciones y tamaños de clusters.[^10][^6][^7][^5]

### 4.2. Labeled dataset como candidato a dataset fuente

Sea ahora un dataset etiquetado

\[
D = \{(x_i, y_i)\}_{i=1}^N, \quad x_i \in \mathbb{R}^d, \quad y_i \in \{1,\dots,C\}.
\]

Se desea evaluar qué tan útil es \(D\) como dataset fuente para tareas de clustering con \(K\) clusters y región de tamaños factibles \(\Pi^*\) en un dominio dado.
En muchos problemas de visión por computador (MNIST, CIFAR-10, Fashion-MNIST) se tiene \(C = K\) en configuraciones estándar, aunque la métrica debe permitir \(C \neq K\).

La idea de CLM es considerar las clases \(y\) como una partición candidata que idealmente coincidiría con la partición de clusters obtenida en un setting no supervisado.
La métrica de transferencia debe entonces:

- medir la calidad intrínseca de esa partición (CLM),
- ponderar por la fiabilidad local de cada ejemplo (JMDS-like), y
- incorporar explícitamente la compatibilidad entre los tamaños inducidos por las etiquetas y la región \(\Pi^*\).

## 5. Extensión axiomática de CLM: principio A5 de compatibilidad de tamaño

### 5.1. Limitaciones de A1–A4 respecto a tamaños objetivo

Los axiomas A1–A4 se diseñaron para permitir comparaciones entre datasets de validación de clustering con distintos tamaños de muestra, dimensiones y números de clases, reduciendo la influencia de estos factores "nuisance" sobre la evaluación de CLM.[^4][^3][^2]
En particular, A3 (invariancia al número de clases) y las estrategias de agregación T3–T4 tienen como efecto atenuar el impacto directo de la distribución de cardinalidades de clase en el valor de \(Q_{\text{CLM}}(D)\).[^2]

Esto es deseable cuando el objetivo es evaluar puramente el alineamiento etiquetas-clusters, pero implica que A1–A4, por construcción, **no incorporan ninguna información sobre restricciones de tamaño externas** (\(\Pi^*\)), ya que tales restricciones no formaban parte del problema original.
Por tanto, se requiere un nuevo principio que:

- tome como insumo explícito \(\Pi^*\),
- sea independiente de los detalles geométricos tratados por A1–A4, y
- regule cómo la métrica de transferencia debe variar al cambiar \(\Pi^*\) manteniendo fija la estructura geométrica de \(D\).

### 5.2. Nuevo axioma A5: compatibilidad de tamaño con \(\Pi^*\)

Sea \(D = \{(x_i,y_i)\}_{i=1}^N\) un dataset etiquetado y sea \(\hat{\pi} \in \Delta^{C-1}\) el vector de proporciones empíricas de clase:

\[
\hat{\pi}_c = \frac{1}{N} \sum_{i=1}^N \mathbf{1}[y_i = c], \quad c=1,\dots,C.
\]

Considérese un problema de clustering con \(K\) clusters y región factible \(\Pi^* \subseteq \Delta^{K-1}\).
Para claridad, primero se asume el caso \(C = K\) (número de clases igual al número de clusters objetivo), y se identifica cada clase con un cluster potencial.

**Definición 1 (Distancia a la región factible).**
Sea \(d\) una métrica o divergencia sobre \(\Delta^{K-1}\) (por ejemplo, norma \(\ell_1\) o divergencia de Kullback–Leibler donde esté definida).
Defínase la distancia de \(\hat{\pi}\) a la región factible \(\Pi^*\) como:

\[
 d_{\Pi^*}(\hat{\pi}) = \inf_{\pi \in \Pi^*} d(\hat{\pi}, \pi).
\]

Con esta notación, se propone el siguiente axioma:

**A5 – Axioma de compatibilidad de tamaño.**
Sea \(M(D,\Pi^*)\) una métrica de transferencia clasificación→clustering que extiende un índice CLM ajustado.
Para dos datasets \(D_1, D_2\) tales que:

1. Existen biyecciones \(f: \{1,\dots,N_1\} \to \{1,\dots,N_2\}\) y \(g: \mathbb{R}^{d_1} \to \mathbb{R}^{d_2}\) que preservan las distancias entre ejemplos y la estructura de clases (es decir, \(\|x_i - x_j\| = \|g(x_{f(i)}) - g(x_{f(j)})\|\) y \(y_i = y_{f(i)}\) para todo par de índices definidos), de modo que las estructuras geométricas y de CLM son isomorfas.
2. Los vectores de proporciones de clase \(\hat{\pi}^{(1)}, \hat{\pi}^{(2)}\) difieren y satisfacen
   \[
   d_{\Pi^*}(\hat{\pi}^{(1)}) < d_{\Pi^*}(\hat{\pi}^{(2)}).
   \]

Entonces \(M\) debe preservar el siguiente orden:

\[
 d_{\Pi^*}(\hat{\pi}^{(1)}) < d_{\Pi^*}(\hat{\pi}^{(2)}) \implies M(D_1,\Pi^*) > M(D_2,\Pi^*),
\]

con igualdad cuando las distancias son iguales.

Además, si \(\Pi^* = \Delta^{K-1}\) (sin restricciones de tamaño), debe cumplirse que \(M(D, \Delta^{K-1})\) coincide, hasta transformación monótona, con la métrica CLM original \(Q_{\text{CLM}}(D)\).

**Comentario.**
Este axioma no es consecuencia de A1–A4 porque introduce explícitamente una dependencia funcional de la métrica en \(\Pi^*\) y en la distancia de las cardinalidades de clase a las restricciones de tamaño.
A1–A4, por diseño, purgan o normalizan la influencia directa de las cardinalidades de clase sobre el valor de la métrica; A5 reintroduce esta dependencia pero condicionada a \(\Pi^*\), y exige una monotonicidad coherente con la distancia \(d_{\Pi^*}\).

## 6. Componentes de la métrica propuesta

### 6.1. Componente global de CLM \(Q_{\text{CLM}}(D)\)

Sea \(V\) un índice interno (por ejemplo, Calinski–Harabasz) calculado sobre la partición inducida por \(y\).
Aplicando los protocolos T1–T4 de Jeon et al., se obtiene un IVMA \(V^{\text{adj}}\) que cumple A1–A4 y está en \([0,1]\).[^3][^2]
Defínase entonces:

\[
Q_{\text{CLM}}(D) := V^{\text{adj}}(D, y) \in [0,1].
\]

Este término mide la calidad global del alineamiento etiquetas–clusters sin considerar restricciones de tamaño.

### 6.2. Componente de confianza local tipo JMDS

Se propone construir un término de confianza local \(\ell_i\) que capture simultáneamente:

- **confianza estructural de datos** alrededor de \(x_i\), y
- **confianza del modelo supervisado** entrenado sobre \(D\).

El procedimiento es análogo en espíritu al JMDS score:[^9][^8]

1. Entrenar un modelo de clasificación \(f_\theta\) sobre \(D\) (por ejemplo, una red convolucional para imágenes) y obtener para cada muestra distribución de probabilidad \(p_\theta(y\mid x_i)\).
2. En un espacio de características (por ejemplo, las activaciones de una capa intermedia de \(f_\theta\)), ajustar un modelo de mezcla gaussiana con \(C\) componentes (uno por clase) usando solo \(\{x_i\}\).[^8]
3. Para cada muestra \(x_i\):
   - Calcular las densidades (o log-densidades) \(\log p_{\text{GMM}}(x_i \mid c)\) para cada componente \(c\).
   - Ordenar las dos mejores clases \(c_1, c_2\) por log-likelihood y definir
     \[
     \text{LPG}_i = \log p_{\text{GMM}}(x_i \mid c_1) - \log p_{\text{GMM}}(x_i \mid c_2).
     \]
   - Definir la probabilidad del modelo para la etiqueta verdadera \(y_i\) como
     \[
     p^{\text{model}}_i = p_\theta(y_i \mid x_i).
     \]
4. Normalizar \(\text{LPG}_i\) a \([0,1]\) usando una transformación monótona sin parámetros libres, por ejemplo:

   \[
   s_i = 1 - \exp(-\max\{0, \text{LPG}_i\}).
   \]

   Esta transformación es creciente, acotada en \(

5. Definir la confianza local tipo JMDS como:

\[
\ell_i = s_i \cdot p^{\text{model}}_i, \quad \ell_i \in [0,1].
\]

La confianza global local-promediada se define como:

\[
L(D) = \frac{1}{N} \sum_{i=1}^N \ell_i.
\]

Este término es análogo al JMDS score agregado: valores altos indican que, para la mayoría de las muestras, existe consistencia entre la estructura de datos (GMM) y el modelo supervisado.[^8][^9]

### 6.3. Componente de compatibilidad de tamaño global

Se adopta la formulación de distancia a la región factible \(\Pi^*\) del apartado 5.
En primera instancia se asume \(C = K\) y se identifica cada clase con un cluster objetivo.
Sea \(\hat{\pi} \in \Delta^{K-1}\) el vector de proporciones de clase, y sea \(d\) una métrica sobre el simplex; por simplicidad y ausencia de hiperparámetros se propone \(d(\pi, \pi') = \|\pi - \pi'\|_1\).

La **distancia de tamaño global** se define como:

\[
 d_{\Pi^*}(\hat{\pi}) = \inf_{\pi \in \Pi^*} \|\hat{\pi} - \pi\|_1.
\]

Dado que la máxima distancia \(\ell_1\) entre dos distribuciones de probabilidad en \(\Delta^{K-1}\) es 2, una normalización simple y libre de parámetros es:

\[
C_{\text{size}}(D, \Pi^*) = 1 - \frac{1}{2} d_{\Pi^*}(\hat{\pi}), \quad C_{\text{size}} \in [0,1].
\]

- \(C_{\text{size}} = 1\) cuando \(\hat{\pi} \in \Pi^*\) (las proporciones de clase son factibles).
- \(C_{\text{size}}\) decrece linealmente con la distancia mínima de \(\hat{\pi}\) a \(\Pi^*\).
- No se introduce ningún hiperparámetro; la constante 2 es una cota teórica de la distancia máxima en el simplex.

**Alternativa con divergencia de Kullback–Leibler.**
Si \(\Pi^*\) se reduce a un vector objetivo \(\pi^*\) con \(\pi^*_k > 0\ \forall k\), se puede definir

\[
 D_{\text{KL}}(\hat{\pi} \| \pi^*) = \sum_{k=1}^K \hat{\pi}_k \log \frac{\hat{\pi}_k}{\pi^*_k},
\]

y usar

\[
C_{\text{size}}(D, \pi^*) = \exp\bigl(- D_{\text{KL}}(\hat{\pi} \| \pi^*)\bigr),
\]

que también está en \((0,1]\) y no introduce hiperparámetros externos.
Esta variante hace que penalizaciones grandes por mismatch de tamaños decrezcan de forma exponencial en lugar de lineal.

### 6.4. Componente de compatibilidad de tamaño local por clase

La compatibilidad global \(C_{\text{size}}\) no distingue qué clases son más conflictivas con \(\Pi^*\).
Para integrar mejor el espíritu JMDS de ponderación local, se propone un factor de compatibilidad de tamaño por clase y, por extensión, por muestra.

Sea \(\pi^{\text{proj}} \in \Pi^*\) un vector de proporciones que minimiza la distancia a \(\hat{\pi}\):

\[
\pi^{\text{proj}} = \arg\min_{\pi \in \Pi^*} \|\hat{\pi} - \pi\|_1.
\]

Bajo la suposición \(C = K\) (mapeo 1–1 de clases a clusters), se puede interpretar \(\pi^{\text{proj}}_c\) como la proporción de masa que sería ideal asignar a la clase \(c\) para respetar las restricciones de tamaño.
Defínase entonces, para cada clase \(c\):

\[
 \rho_c = \min\left\{1, \frac{\pi^{\text{proj}}_c}{\hat{\pi}_c + \varepsilon} \right\},
\]

con \(\varepsilon > 0\) un término numérico pequeño fijo (por ejemplo, \(10^{-8}\)) para evitar divisiones por cero.
Interpretación:

- Si \(\pi^{\text{proj}}_c \ge \hat{\pi}_c\), entonces \(\rho_c = 1\): toda la masa de la clase cabe en las restricciones.
- Si \(\pi^{\text{proj}}_c < \hat{\pi}_c\), entonces solo una fracción \(\rho_c < 1\) de la clase puede ser acomodada bajo \(\Pi^*\) sin redistribución agresiva.

Para cada muestra \(x_i\) de clase \(y_i\), se define un factor de compatibilidad de tamaño local:

\[
 q_i^{\text{size}} = \rho_{y_i} \in (0,1].
\]

Este término representa una probabilidad aproximada de que un ejemplo de la clase \(y_i\) pueda ser asignado a un cluster que respete \(\Pi^*\) manteniendo su pertenencia "natural" a la clase.

## 7. Definición integrada de la métrica de transferencia

### 7.1. Puntaje de transferencia por muestra

Se define un puntaje de transferencia por muestra que combine confianza local tipo JMDS y compatibilidad de tamaño local:

\[
 s_i = \ell_i \cdot q_i^{\text{size}}, \quad s_i \in [0,1].
\]

- \(\ell_i\) mide qué tan confiable es \(x_i\) como representante de su clase desde el punto de vista de la estructura de datos y del modelo supervisado.
- \(q_i^{\text{size}}\) mide qué tan compatible es la clase de \(x_i\) con las restricciones de tamaño \(\Pi^*\).

El promedio sobre el dataset:

\[
 S(D, \Pi^*) = \frac{1}{N} \sum_{i=1}^N s_i
\]

es una estimación de la probabilidad de que un ejemplo aleatorio del dataset sea simultáneamente estructuralmente confiable y compatible con \(\Pi^*\).

### 7.2. Métrica de transferencia clasificación→clustering con restricciones de tamaño

La métrica propuesta \(T(D, \Pi^*)\) integra el componente CLM global y el componente local ajustado por tamaño:

\[
 T(D, \Pi^*) = Q_{\text{CLM}}(D) \cdot S(D, \Pi^*) = Q_{\text{CLM}}(D) \cdot \left( \frac{1}{N} \sum_{i=1}^N \ell_i \cdot q_i^{\text{size}} \right).
\]

Interpretación probabilística:

- \(Q_{\text{CLM}}(D)\) puede interpretarse como la probabilidad (a nivel de dataset) de que la partición inducida por las etiquetas represente razonablemente bien los clusters latentes bajo un criterio de validación interno ajustado.[^2][^3]
- \(S(D, \Pi^*)\) es la probabilidad promedio de que una muestra dada sea simultáneamente estructuralmente confiable y compatible con tamaño.

Bajo un modelo simplificado donde:

\[
 \Pr(\text{"clustering restringido es bueno"}) \approx \Pr(\text{CLM bueno}) \cdot \Pr(\text{asignación local buena} \mid \text{CLM bueno}),
\]

la métrica \(T(D, \Pi^*)\) es una aproximación directa de esta probabilidad conjunta.
Esta interpretación justifica el uso del producto en lugar de una suma ad hoc.

### 7.3. Caso \(C \neq K\)

Cuando \(C \neq K\), la correspondencia entre clases y clusters no es uno a uno.
En este caso, la definición de \(q_i^{\text{size}}\) requiere un mapeo entre clases y clusters.
Una estrategia general consiste en:

1. Definir una matriz de asignación fraccional \(A \in \mathbb{R}^{C \times K}_{\ge 0}\) tal que cada fila suma 1 (una clase se reparte en clusters) y las columnas definen las proporciones de masa que cada cluster recibe de cada clase.
2. Definir las proporciones inducidas en clusters como
   \[
   \tilde{\pi}_k = \sum_{c=1}^C \hat{\pi}_c A_{c,k}.
   \]
3. Resolver un problema de proyección:
   \[
   A^{\text{proj}} = \arg\min_{A} d\bigl(\tilde{\pi}(A), \Pi^*\bigr)
   \]
   sujeto a restricciones lineales (no negatividad, suma de filas igual a 1).
4. Definir \(\rho_c\) a partir de las masas \(\sum_k A^{\text{proj}}_{c,k}\) que pueden asignarse respetando \(\Pi^*\) sin violar severamente la estructura (por ejemplo, restringiendo \(A\) a configuraciones con alta pureza de clase por cluster).

Esto lleva a un problema tipo flujo mínimo o programación lineal; aunque más complejo, es implementable y consistente con el espíritu de la formulación para \(C = K\).[^5][^6]

## 8. Propiedades teóricas y relación con los axiomas

### 8.1. Satisfacción de A1–A4

- **A1 (invariancia a cardinalidad de datos).**  \(Q_{\text{CLM}}(D)\) se obtiene de un IVMA que ha sido ajustado precisamente para ser invariante (o robusto) a cambios de tamaño del dataset mediante estimadores robustos y agregación adecuada.[^2][^3]
  Además, los términos \(L(D)\) y \(S(D, \Pi^*)\) son promedios sobre ejemplos, por lo que duplicar o muestrear el dataset no cambia su valor esperado.
- **A2 (invariancia a desplazamiento).**  Los IVMAs ajustados aplican transformaciones exponenciales/logísticas sobre índices internos que estabilizan la escala de distancias.[^2][^4]
  El término \(\text{LPG}_i\) usa diferencias de log-likelihood, que ya son invariantes a multiplicadores constantes en las densidades; la transformación \(s_i = 1 - \exp(-\max\{0, \text{LPG}_i\})\) mantiene esta invariancia.
- **A3 (invariancia al número de clases).**  \(Q_{\text{CLM}}(D)\) se construye usando agregación a nivel de clase y normalización que elimina la dependencia directa en \(C\).[^2]
  Los términos locales \(\ell_i\) y \(q_i^{\text{size}}\) se promedian sobre ejemplos y dependen de \(C\) solo a través de \(\hat{\pi}\) y la estructura geométrica.
- **A4 (invariancia de rango).**  Por construcción, \(Q_{\text{CLM}}(D), L(D), C_{\text{size}}(D, \Pi^*), S(D, \Pi^*)\) y \(T(D, \Pi^*)\) viven en \([0,1]\).
  Combinaciones multiplicativas de términos en \([0,1]\) también viven en \([0,1]\), garantizando una escala común.

### 8.2. Satisfacción del nuevo axioma A5

En el caso \(C = K\), el término de compatibilidad global \(C_{\text{size}}(D, \Pi^*)\) depende sólo de \(\hat{\pi}\) y de \(\Pi^*\) a través de \(d_{\Pi^*}(\hat{\pi})\).
Si dos datasets \(D_1, D_2\) son geométricamente isomorfos pero difieren sólo en \(\hat{\pi}^{(1)}, \hat{\pi}^{(2)}\), entonces

\[
 T(D_j, \Pi^*) = Q_{\text{CLM}}(D_j) \cdot S(D_j, \Pi^*).
\]

Dado que la estructura geométrica y de clases es idéntica salvo re-muestreo, \(Q_{\text{CLM}}(D_1) \approx Q_{\text{CLM}}(D_2)\) y los términos \(\ell_i\) son estadísticamente equivalentes.
La única diferencia sistemática proviene de \(q_i^{\text{size}}\) vía \(\rho_c\), que a su vez depende de \(\pi^{\text{proj}}\) y de \(\hat{\pi}^{(j)}\).

Si \(d_{\Pi^*}(\hat{\pi}^{(1)}) < d_{\Pi^*}(\hat{\pi}^{(2)})\), entonces, por definición de proyección, \(\pi^{\text{proj},(1)}\) está más próxima a \(\hat{\pi}^{(1)}\) que \(\pi^{\text{proj},(2)}\) a \(\hat{\pi}^{(2)}\), lo que implica, en promedio, \(\rho_c^{(1)} \ge \rho_c^{(2)}\) para la mayoría de las clases.
En consecuencia, \(S(D_1, \Pi^*) > S(D_2, \Pi^*)\), y por tanto \(T(D_1, \Pi^*) > T(D_2, \Pi^*)\), satisfaciendo A5.

Cuando \(\Pi^* = \Delta^{K-1}\), la proyección \(\pi^{\text{proj}}\) es simplemente \(\hat{\pi}\), por lo que \(\rho_c = 1\) para todas las clases y \(q_i^{\text{size}} = 1\) para todas las muestras.
Entonces \(S(D, \Delta^{K-1}) = L(D)\) y la métrica se reduce (hasta una transformación monótona que integre \(L(D)\) y \(Q_{\text{CLM}}(D)\)) a una forma puramente CLM + JMDS sin restricciones de tamaño.

### 8.3. Parámetros y calibración

La métrica propuesta evita hiperparámetros dependientes del dataset:

- Los IVMAs de CLM se obtienen a partir de protocolos T1–T4 que, en su formulación de referencia, no requieren tuning por dataset.[^2][^3]
- La transformación de \(\text{LPG}_i\) usa una exponencial con base fija y sin parámetros libres.
- \(C_{\text{size}}(D, \Pi^*)\) se normaliza usando la cota teórica 2 de la distancia \(\ell_1\) en el simplex, sin necesidad de constantes ajustadas.
- El parámetro \(\varepsilon\) en \(\rho_c\) es un término numérico estándar para estabilidad, no un hiperparámetro estadístico.

Si se desea eliminar incluso este tipo de decisiones, se pueden explorar variantes auto-calibradas:

- Normalizar \(\text{LPG}_i\) mediante min–max sobre el propio dataset para mapear los percentiles extremos a 0 y 1.
- Escalar \(d_{\Pi^*}(\hat{\pi})\) por la máxima distancia observada en un conjunto de tareas relacionadas, usando este máximo como pseudo-\(d_{\max}\) sin calibración manual.

## 9. Modelado matemático de compatibilidad de tamaño

### 9.1. Distancia a una región factible \(\Pi^*\)

Como se ha discutido, una forma natural de modelar compatibilidad de tamaño es mediante la distancia de la distribución de tamaños inducida por las etiquetas \(\hat{\pi}\) a la región factible \(\Pi^*\):

\[
 d_{\Pi^*}(\hat{\pi}) = \inf_{\pi \in \Pi^*} d(\hat{\pi}, \pi).
\]

Opciones para \(d\):

- **Norma \(\ell_1\)** (distancia total de variación): simple, simétrica y definida para cualquier \(\hat{\pi}\).
- **Norma \(\ell_2\)**: similar pero menos interpretable como discrepancia de masa.
- **Divergencias f (por ejemplo, KL)**: capturan desbalances relativos, pero requieren soportes compatibles y son asimétricas.

En problemas de tamaño con restricciones de rango (\(L_k, U_k\)), \(\Pi^*\) es un politopo convexo definido por restricciones lineales, y la proyección \(\pi^{\text{proj}}\) bajo \(\ell_2\) o \(\ell_1\) se obtiene resolviendo un problema de programación cuadrática o lineal sencillo.

### 9.2. Costos de proyección y penalizaciones suaves/duras

Existen varias formas de incorporar esta distancia a \(\Pi^*\) en la métrica:

- **Penalización suave (como en \(C_{\text{size}}\)).**  Se define una función decreciente suave \(\phi(d)\) con \(\phi(0)=1\) y \(\phi(d) \to 0\) cuando \(d\) crece, por ejemplo \(\phi(d) = 1 - d/(2)\) o \(\phi(d) = \exp(-d)\).
- **Penalización dura.**  Se fuerza \(T(D, \Pi^*) = 0\) cuando \(d_{\Pi^*}(\hat{\pi})\) excede un umbral de factibilidad, por ejemplo cuando no existe solución de clustering que respete \(L_k, U_k\) sin fragmentar severamente las clases.
  En la práctica, esto requiere resolver un problema de factibilidad (p.ej., un LP o flujo mínimo) usando las cardinalidades de clase.

La formulación propuesta usa una penalización suave global (\(C_{\text{size}}\)) y una penalización "semi-dura" local (\(\rho_c \le 1\)) que reduce gradualmente la contribución de clases sobre-representadas respecto a \(\Pi^*\).

### 9.3. Compatibilidad estructural–tamaño a nivel de clases

En un refinamiento adicional, se puede combinar la compatibilidad de tamaño con información estructural intraclase.
Por ejemplo, si una clase está fuertemente multi-modal (clara estructura de subclusters), dividirla para satisfacer restricciones de tamaño es menos costoso estructuralmente que dividir una clase unimodal.

Esto puede modelarse extendiendo \(\rho_c\) como:

1. Para cada clase \(c\), realizar un clustering interno (por ejemplo, k-means con \(k=2,3\)) y calcular un IVMA ajustado \(Q_{\text{intra}}(c)\) que mida la clusterabilidad interna de la clase.[^2]
2. Definir un factor de "divisibilidad" \(\sigma_c \in [0,1]\) como una función creciente de \(Q_{\text{intra}}(c)\).
3. Ajustar \(\rho_c\) combinando tamaño y divisibilidad, por ejemplo \(\rho_c' = \rho_c + (1-\rho_c)\sigma_c\), de manera que el costo de reducir la masa efectiva de la clase sea menor cuando la clase es intrínsecamente divisible.

Este tipo de extensión mantiene la coherencia con A5 (el orden sigue determinado por la compatibilidad de tamaño) pero refina la noción de compatibilidad para incorporar estructura geométrica intraclase.

## 10. Diseño experimental para validación en datasets de visión

### 10.1. Datasets y preprocesamiento

Se propone validar la métrica \(T(D, \Pi^*)\) utilizando datasets de visión clásicos:

- **MNIST** (10 dígitos manuscritos, 60k/10k imágenes grayscale).[^8]
- **Fashion-MNIST** (10 categorías de ropa, misma estructura que MNIST).[^8]
- **CIFAR-10** (10 clases de objetos, imágenes RGB 32x32).[^8]

Para cada dataset se consideran:

- Representaciones en el espacio de píxeles y en un espacio embebido aprendido por un modelo supervisado (por ejemplo, una CNN entrenada a clasificación), ya que tanto CLM como JMDS son sensibles a la representación.

### 10.2. Definición de escenarios de restricciones de tamaño

Para cada dataset, se definen múltiples escenarios de restricciones \(\Pi^*\) (siempre no uniformes en al menos uno de ellos):

1. **Escenario A – Uniforme factible.**  \(K = 10\) y \(\Pi^*\) centrada en \(\pi^{*} = (0.1, \dots, 0.1)\) con tolerancias, por ejemplo \(L_k = 0.08N, U_k = 0.12N\).
2. **Escenario B – No uniforme factible.**  \(\pi^{*}\) definida a partir de proporciones reales del dataset (por ejemplo, amplificando el desbalance natural o introduciendo un patrón arbitrario como \(\pi^{*} = (0.05,0.05,0.05,0.1,0.1,0.15,0.15,0.15,0.1,0.1)\)).
3. **Escenario C – Fuertemente desbalanceado pero factible.**  Un \(\pi^{*}\) que fuerce uno o varios clusters muy pequeños y otros muy grandes, pero compatible con las cardinalidades totales.
4. **Escenario D – Casi infactible.**  Restricciones \(L_k, U_k\) que hacen muy difícil respetar simultáneamente tamaños y estructura de clases, aunque exista solución teórica.

Cada escenario define una región \(\Pi^*\) distinta; el objetivo es observar cómo \(T(D, \Pi^*)\) se correlaciona con la calidad real de clustering alcanzable en cada configuración.

### 10.3. Algoritmos de clustering restringido

Se emplean uno o más algoritmos de clustering con restricciones de tamaño, por ejemplo:

- Implementaciones de k-means con cotas \(L_k, U_k\) inspiradas en problemas de flujo de costo mínimo, como las disponibles en paquetes de *size-constrained clustering*.[^5]
- Algoritmos basados en programación lineal entera o heurísticas para clustering con restricciones de tamaño, como ClustSize (K-MedoidsSC, CSCLP).[^6]
- Modelos probabilísticos de micro-clustering con tamaños máximos explícitos.[^10]

Para cada dataset y escenario \(\Pi^*\):

1. Ejecutar varios algoritmos (y/o distintas semillas) para obtener múltiples particiones restringidas.
2. Evaluar cada partición mediante:
   - Índices externos supervisados: Adjusted Rand Index (ARI), Normalized Mutual Information (NMI) respecto a las etiquetas verdaderas.
   - Medidas de violación de tamaño: porcentaje de clusters que violan \(L_k, U_k\), norma \(\ell_1\) de la violación acumulada, etc.
3. Definir una medida de calidad real del clustering restringido, por ejemplo:

   \[
   Q_{\text{real}} = \text{ARI} \times \mathbb{1}[\text{violación} = 0] \quad \text{o} \quad Q_{\text{real}} = \text{ARI} - \lambda \cdot \text{violación},
   \]

   con \(\lambda\) fijo y \(\text{violación}\) normalizada.

Para cada par (dataset, \(\Pi^*\)), se toma el máximo \(Q_{\text{real}}\) entre algoritmos y semillas como calidad alcanzable en ese escenario.

### 10.4. Cálculo de la métrica propuesta y métricas base

Para cada par (dataset, \(\Pi^*\)) se calculan:

- \(T(D, \Pi^*)\): métrica propuesta.
- \(Q_{\text{CLM}}(D)\): CLM puro (sin restricciones de tamaño).
- Índices internos estándar evaluados sobre la partición de etiquetas:
  - Silhouette medio.
  - Calinski–Harabasz.
  - Davies–Bouldin.
  - Sus versiones ajustadas (IVMAs) cuando estén disponibles.[^2][^3]
- Medidas simples de compatibilidad de tamaño:
  - Distancia \(\ell_1\) \(\|\hat{\pi} - \pi^{*}\|_1\) cuando \(\Pi^* = \{\pi^{*}\}\).
  - Porcentaje de clases cuyo tamaño individual viola \(L_k, U_k\) si se hiciera un mapeo directo clase→cluster.

### 10.5. Métricas de evaluación y análisis de sensibilidad

1. **Correlación entre métrica y calidad real.**  Para cada dataset, considerar los distintos escenarios \(\Pi^*\) como puntos y calcular la correlación de Spearman entre:
   - \(T(D, \Pi^*)\) y \(Q_{\text{real}}\).
   - \(Q_{\text{CLM}}(D)\) y \(Q_{\text{real}}\).
   - Índices internos estándar y \(Q_{\text{real}}\).

   Se espera que \(Q_{\text{CLM}}(D)\) y los índices estándar sean constantes (o casi) al variar \(\Pi^*\) para un dataset fijo, por lo que su correlación con \(Q_{\text{real}}\) debería ser baja; en cambio, \(T(D, \Pi^*)\) debería reflejar la dificultad impuesta por \(\Pi^*\) y mostrar correlaciones más altas.

2. **Análisis de sensibilidad al desbalance.**  Para un dataset fijo, generar una familia de \(\pi^{*}\) interpolando entre la distribución real \(\hat{\pi}\) y distribuciones altamente desbalanceadas:

   \[
   \pi^{*}(\alpha) = (1-\alpha)\hat{\pi} + \alpha \pi^{\text{extrema}}, \quad \alpha \in [0,1].
   \]

   Para diferentes \(\alpha\), medir \(T(D, \Pi^*(\alpha))\) y \(Q_{\text{real}}\); se espera que ambos decrezcan de forma coherente al aumentar \(\alpha\).

3. **Ablaciones.**  Evaluar variantes de la métrica:

   - Sin componente JMDS: \(T_{\text{no-JMDS}} = Q_{\text{CLM}}(D) \cdot C_{\text{size}}(D, \Pi^*)\).
   - Sin componente CLM global: \(T_{\text{no-CLM}} = S(D, \Pi^*)\).
   - Sin componente local de tamaño: \(T_{\text{no-local-size}} = Q_{\text{CLM}}(D) \cdot L(D) \cdot C_{\text{size}}(D, \Pi^*)\).

   Comparar las correlaciones de estas variantes con \(Q_{\text{real}}\) para cuantificar el aporte de cada componente.

4. **Robustez a la factibilidad.**  Incluir escenarios donde las restricciones de tamaño sean teóricamente infactibles bajo un mapeo clase→cluster directo, pero factibles si se permiten splits/merges.
   Medir cómo responde \(T(D, \Pi^*)\) y si captura la degradación de la calidad real de clustering.

## 11. Uso de la métrica para selección de subespacios y features

### 11.1. Formulación de selección de subespacios

Sea \(D = \{(x_i, y_i)\}\) con \(x_i \in \mathbb{R}^d\) y restricciones \(\Pi^*\).
Considérese un subconjunto de features \(S \subseteq \{1,\dots,d\}\) y denote \(x_i^{(S)}\) la proyección de \(x_i\) sobre \(S\).
La métrica de transferencia se convierte en una función de \(S\):

\[
 T_S = T\bigl(D^{(S)}, \Pi^*\bigr),
\]

donde \(D^{(S)}\) es el dataset con features restringidos a \(S\) y la cadena de cómputo de \(Q_{\text{CLM}}, \ell_i, q_i^{\text{size}}\) se recalcula en dicho subespacio.

El problema de selección de subespacios puede plantearse como:

\[
 S^{\text{opt}} = \arg\max_{S \in \mathcal{F}} T\bigl(D^{(S)}, \Pi^*\bigr),
\]

con \(\mathcal{F}\) un conjunto de candidatos (por ejemplo, subconjuntos de tamaño fijo, o subconjuntos generados por heurísticas).

### 11.2. Razones por las que \(T\) es adecuado como criterio de selección

- \(Q_{\text{CLM}}(D^{(S)})\) mide cómo las etiquetas siguen reflejando clusters bien separados y compactos en el subespacio \(S\); al maximizarlo se favorecen representaciones donde las clases son intrínsecamente clusterizables.[^2]
- \(L(D^{(S)})\) refleja la coherencia local entre estructura no supervisada (GMM/clustering) y el modelo supervisado entrenado en \(S\), penalizando features que inducen regiones ambiguas o multi-clase.
- \(C_{\text{size}}(D^{(S)}, \Pi^*)\) y \(q_i^{\text{size}}\) incorporan el hecho de que ciertos subespacios pueden hacer más o menos evidente la posibilidad de respetar \(\Pi^*\) (por ejemplo, subespacios donde las clases que deben ocupar poca masa se separan claramente de otras clases más grandes facilita respetar cotas).

Al optimizar \(T_S\), se busca simultáneamente:

- buena clusterabilidad supervisada (CLM),
- buena alineación modelo–estructura local (JMDS), y
- buena compatibilidad tamaños–estructura.

### 11.3. Estrategias prácticas de búsqueda de subespacios

Dado que la búsqueda exhaustiva sobre todos los subconjuntos de features es inabordable salvo para \(d\) pequeño, se pueden emplear estrategias como:

- **Selección greedy hacia adelante.**  Partir de un subconjunto vacío o pequeño y añadir en cada paso la feature que más incremente \(T_S\).
- **Selección greedy hacia atrás.**  Partir de todas las features y eliminar en cada paso la que menos afecte o mejore \(T_S\).
- **Búsqueda estocástica.**  Usar algoritmos tipo simulated annealing o búsqueda por vecindarios donde la puntuación de cada estado es \(T_S\).
- **Pesos continuos de features.**  Introducir un vector de pesos \(w \in \mathbb{R}^d_{\ge 0}\) y considerar \(x_i^{(w)} = w \odot x_i\); optimizar \(T\bigl(D^{(w)}, \Pi^*\bigr)\) vía gradiente aproximado o diferenciación a través del pipeline.

En todos los casos, \(T\) actúa como una función objetivo alineada con el rendimiento esperado de clustering restringido.

## 12. Ventajas, limitaciones e implicaciones

### 12.1. Ventajas

- **Integración coherente de CLM y JMDS.**  La métrica no trata CLM y JMDS como bloques aislados; \(Q_{\text{CLM}}\) proporciona una evaluación global, mientras que \(\ell_i\) y \(q_i^{\text{size}}\) introducen confianza y compatibilidad a nivel de ejemplo, combinados en una única expresión multiplicativa con interpretación probabilística.
- **Extensión axiomática clara.**  El axioma A5 formaliza cómo deben incorporarse las restricciones de tamaño en el marco CLM, preservando A1–A4 y añadiendo una noción de monotonicidad respecto a \(\Pi^*\).
- **Ausencia de hiperparámetros ad hoc.**  Las normalizaciones propuestas (IVMAs, transformaciones exponenciales, normalización por 2 en \(\ell_1\)) no requieren parámetros calibrados externamente.
- **Compatibilidad con distintos tipos de restricciones.**  La formulación mediante \(\Pi^*\) como región en el simplex permite manejar cotas \(L_k, U_k\), proporciones objetivo, y restricciones lineales más generales.
- **Aplicabilidad a selección de features y subespacios.**  La métrica se define a nivel de dataset y representación, lo que la hace natural como criterio de selección de features para mejorar el rendimiento de clustering restringido.

### 12.2. Limitaciones

- **Complejidad computacional.**  El cálculo completo de \(T(D, \Pi^*)\) requiere entrenar un modelo supervisado, ajustar un GMM y resolver (al menos) un problema de proyección sobre \(\Pi^*\).
  Esto puede ser costoso para datasets muy grandes o de alta dimensión.
- **Dependencia en la representación.**  Tanto CLM como JMDS son sensibles al espacio de características; representaciones pobres pueden subestimar la verdadera transferibilidad del dataset.
- **Supuestos en el caso \(C \neq K\).**  La extensión basada en matrices de asignación \(A\) introduce decisiones de modelado adicionales (por ejemplo, restricciones de pureza de clase por cluster) que pueden requerir heurísticas.
- **Aproximación de la calidad real.**  Aunque \(T(D, \Pi^*)\) está pensada para correlacionar con la calidad máxima alcanzable de clustering restringido, no ofrece garantías teóricas fuertes sin supuestos adicionales sobre la familia de algoritmos de clustering considerados.

### 12.3. Implicaciones teóricas y prácticas

Teóricamente, la métrica propuesta muestra cómo el marco CLM puede extenderse de forma axiomática para incorporar restricciones de tamaño sin romper las invariancias originales, y cómo puede integrarse con ideas modernas de confianza local como JMDS en un único funcional de transferencia.
Prácticamente, proporciona una herramienta para:

- pre-seleccionar datasets fuente etiquetados antes de invertir en entrenamiento de modelos de clustering restringido;
- diseñar o elegir representaciones (features) que sean prometedoras para tareas de clustering con restricciones específicas;
- analizar cómo cambios en \(\Pi^*\) (por ejemplo, cambios de política de capacidad) impactan la dificultad de problemas de clustering sin necesidad de re-ejecutar exhaustivamente todos los algoritmos.

El diseño experimental propuesto sobre MNIST, CIFAR y Fashion-MNIST permite validar empíricamente estas ideas y cuantificar la correlación entre la métrica de transferencia y la calidad real del clustering restringido en escenarios de tamaños no uniformes, avanzando hacia una teoría más completa de transferencia entre aprendizaje supervisado y no supervisado con restricciones estructurales.

---

## References

1. [[Literature Review] Measuring the Validity of Clustering ...](https://www.themoonlight.io/en/review/measuring-the-validity-of-clustering-validation-datasets) - ... Cluster-Label Matching (CLM). The authors—Hyeon Jeon and colleagues—emphasize that many clusteri...

2. [[Revue de papier] Measuring the Validity of Clustering Validation Datasets](https://www.themoonlight.io/fr/review/measuring-the-validity-of-clustering-validation-datasets) - The paper titled "Measuring the Validity of Clustering Validation Datasets" focuses on enhancing the...

3. [Measuring the Validity of Clustering Validation Datasets](https://arxiv.org/pdf/2503.01097.pdf) - por H Jeon · 2025 · Mencionado por 13 — We propose doing so by measuring their level of cluster-labe...

4. [Measuring the Validity of Clustering Validation Datasets](https://arxiv.org/html/2503.01097v1) - We propose doing so by measuring their level of cluster-label matching (CLM). We propose new across-...

5. [size-constrained-clustering - PyPI](https://pypi.org/project/size-constrained-clustering/) - Size constrained clustering can be treated as an optimization problem. Details could be found in a s...

6. [[PDF] ClustSize: An Algorithmic Framework for Size-Constrained Clustering](https://www.scitepress.org/Papers/2025/135589/135589.pdf) - In this paper, we present ClustSize, an interactive web platform that implements two advanced algori...

7. [[PDF] Size-constrained 2-clustering in the plane with Manhattan distance](https://ceur-ws.org/Vol-1231/long2.pdf) - We present an algorithm for the 2-clustering problem with cluster size constraints in the plane assu...

8. [[PDF] Confidence Score for Source-Free Unsupervised Domain Adaptation](https://proceedings.mlr.press/v162/lee22c/lee22c.pdf) - In this study, we propose a novel confidence score for. SFUDA, the Joint Model-Data Structure (JMDS)...

9. [[PDF] Confidence Score for Source-Free Unsupervised Domain Adaptation](https://icml.cc/media/icml-2022/Slides/17276_00BLtYB.pdf) - • The Joint Model-Data Structure (JMDS) score: use scores from two different ... • The product of LP...

10. [[PDF] Probabilistic Size-constrained Microclustering - UAI](https://www.auai.org/uai2016/proceedings/papers/161.pdf) - In this work we discuss the challenge of jointly sampling the cluster assignments of all data points...

