Diseño de una Métrica de Transferencia para la Utilización de Conjuntos de Datos de Clasificación en Tareas de Clustering

1. Extensión Teórica: El Axioma de Compatibilidad de Restricción (A5)

El marco axiomático original de CLM establece que una métrica de validación debe ser invariante a la cardinalidad de los datos (A1), al desplazamiento en altas dimensiones (A2), al número de clases (A3) y mantener un rango acotado (A4).

Sin embargo, estos axiomas asumen implícitamente que un clustering es "bueno" si refleja la estructura topológica natural de los datos. En un entorno con restricciones de tamaño (size-constrained clustering), la topología natural puede entrar en conflicto directo con las proporciones exigidas por la tarea.

Por lo tanto, propongo introducir un quinto axioma (A5) para tu tesis:

Axioma A5: Invariancia y Penalización por Factibilidad de Restricción (Constraint-Feasibility Compatibility)

Una métrica de transferencia $`f(C, X, \delta, \Pi^*)`$ evaluada sobre un conjunto de datos $`X`$ con partición natural $`C`$ y una región de tamaños factibles $`\Pi^*`$, debe penalizar estrictamente la divergencia entre la distribución empírica de clases $`\pi`$ y el punto óptimo proyectado $`\hat{\pi} \in \Pi^*`$. Además, la magnitud de esta penalización debe ser proporcional a la certeza estructural de las muestras que deben ser reasignadas para satisfacer $`\Pi^*`$.

Justificación de A5: A1-A4 no cubren esto porque evalúan la calidad intrínseca del etiquetado asumiendo clusters libres. Si un algoritmo debe forzar proporciones $\Pi^*$, inevitablemente romperá fronteras naturales. Un dataset es apto para transferencia solo si la masa de datos que debe moverse para satisfacer la restricción habita en regiones de baja confianza estructural.

2. Diseño de la Métrica de Transferencia Acoplada ($T_{C \to C}$)

Para evitar depender de factores externos calibrados empíricamente y para integrar JMDS y CLM de forma coherente, propongo que la métrica evalúe el Costo Esperado de Cumplimiento de Restricciones.

2.1. Confianza Local a nivel de Muestra (Inspiración JMDS)

Primero, evaluamos cada muestra $x_i$ usando una lógica análoga a JMDS (que combina la estructura de datos LPG y el conocimiento del modelo MPPL). Como tenemos etiquetas reales (no pseudo-etiquetas), definimos la confianza de la muestra $\gamma(x_i) \in [0,1]$ como:

$$\gamma(x_i) = \text{LPG}(x_i) \times P_{struct}(y_i | x_i)$$

LPG (Log-Probability Gap Estructural): Utilizando una estimación de densidad local (KDE o GMM sobre el espacio de características), es la diferencia normalizada entre la log-probabilidad de pertenecer a su clase real frente a la clase competidora más cercana. Captura el grado de separación en la frontera.

$P_{struct}(y_i | x_i)$: La probabilidad de consistencia local (por ejemplo, la proporción de k-vecinos más cercanos que comparten la misma etiqueta $y_i$).

Muestras en el núcleo profundo del cluster tendrán $\gamma(x_i) \approx 1$. Muestras en fronteras ambiguas tendrán $\gamma(x_i) \approx 0$.

2.2. Penalización Macroscópica por Restricciones de Tamaño

Sean $\pi = (\pi_1, \dots, \pi_K)$ las proporciones naturales de las clases en el dataset fuente. Sea $\Pi^*$ el espacio factible definido por las restricciones del usuario (por ejemplo, $L_k \le n_k \le U_k$).

Encontramos la distribución objetivo más cercana $\hat{\pi} \in \Pi^*$ minimizando el costo de transporte óptimo (Wasserstein) o la divergencia KL:

$$\hat{\pi} = \arg\min_{p \in \Pi^*} D_{KL}(\pi || p)$$

El exceso de masa que debe ser desplazado (reasignado) del cluster $k$ es el ratio de desbordamiento:

$$\Delta_k = \max(0, \pi_k - \hat{\pi}_k)$$

2.3. Integración Coherente (Métrica Final Auto-calibrada)

Aquí es donde la magia ocurre. Si el algoritmo de clustering tiene que quitar $\Delta_k$ proporción de datos de la clase $k$, lo ideal es que quite los datos más ambiguos (menor $\gamma(x_i)$).

Si ordenamos las muestras de la clase $k$ de menor a mayor confianza $\gamma$, el Costo de Reasignación Estructural para esa clase es la integral de la confianza de las muestras que inevitablemente serán reasignadas. Sea $F_k^{-1}$ la función de distribución inversa (cuantil) empírica de las confianzas $\gamma$ dentro de la clase $k$. La penalización de tamaño para la clase $k$ es:

$$P_{size}(k) = \frac{1}{\Delta_k} \int_{1-\Delta_k}^{1} F_k^{-1}(q) dq$$

(Si un cluster debe ceder el 10% de su masa ($\Delta_k = 0.1$), $P_{size}$ mide la confianza media del 10% de las muestras menos seguras. Si incluso los puntos marginales tienen alta confianza, significa que romper el cluster destruirá topología fuerte, lo que resulta en una penalización alta).

La métrica de transferencia global $T_{C \to C}$ se formula combinando una evaluación estructural base (como $CH_A$ que cumple A1-A4) descontando el costo de las restricciones (A5):

$$T_{C \to C} = \text{IVM}_A(X, C) \times \left( 1 - \sum_{k=1}^K \omega_k P_{size}(k) \right)$$

Donde $\omega_k = \Delta_k / \sum_j \Delta_j$. Esta métrica está inherentemente acotada en $[0,1]$ asumiendo que el $\text{IVM}_A$ esté normalizado, no requiere calibración de hiperparámetros de escala, y fusiona la topología microscópica (JMDS) con la restricción macroscópica (CLM/A5).

3. Diseño Experimental Sugerido

Para validar la métrica rigurosamente sin caer en la trampa del k-means balanceado:

1. Preparación de Datasets (Ej: CIFAR-10, Fashion-MNIST):

Extrae embeddings usando una red preentrenada (ej. ResNet para CIFAR-10). Crea múltiples sub-versiones (escenarios) de los datasets inyectando desequilibrios intencionales (ej. Clase 1 = 50% de los datos, Clase 2 = 5%).

2. Definición de Restricciones $\Pi^*$ (Variables Independientes):

Escenario A (Relajado): Bounds amplios que incluyen las proporciones naturales.

Escenario B (Estricto y Conflictivo): Exigir particiones uniformes (10% por clase) sobre un dataset naturalmente desbalanceado.

Escenario C (Sesgado): Exigir que una clase específica contenga el 40% de los datos cuando naturalmente tiene el 10%.

3. Baselines de Validación:

Compara $T_{C \to C}$ contra CH, Silhouette, Davies-Bouldin, y las versiones ajustadas puras propuestas por Jeon et al. ($CH_A$).

4. Ejecución y Medición (Variable Dependiente):

Aplica algoritmos de Size-Constrained Clustering (como Constrained K-Means de Bradley et al. o enfoques basados en Optimal Transport). Calcula métricas externas de éxito (Adjusted Rand Index o NMI) comparando el output restringido contra las etiquetas reales.

5. Análisis de Correlación:

Demuestra mediante correlación de rango de Spearman que $T_{C \to C}$ predice el NMI final mucho mejor que los índices estándar, especialmente en los escenarios conflictivos (B y C), demostrando que la penalización por reasignación de alta confianza captura la verdadera "transferibilidad".

4. Aplicación: Selección de Subespacios / Feature Selection

Tu métrica no solo sirve para evaluación, sino como función objetivo para mejorar el espacio latente antes del clustering.

Dado que $T_{C \to C}$ evalúa qué tan doloroso es imponer la restricción de tamaño, podemos utilizar un vector de pesos binario o continuo $w \in [0,1]^D$ sobre las características y resolver:

$$w^* = \arg\max_{w} T_{C \to C}(X \odot w, C, \Pi^*)$$

Al optimizar esto (mediante búsqueda voraz, algoritmos genéticos, o descenso de gradiente si relajas la métrica para hacerla diferenciable), el modelo seleccionará subespacios (features) donde las fronteras entre clases en las regiones que deben ser reasignadas sean lo más difusas y superpuestas posible, disminuyendo $P_{size}(k)$.

En la práctica, esto significa que el feature selection deforma el espacio para que los clusters naturalmente absorban las restricciones de tamaño sin sacrificar la coherencia de su núcleo denso.