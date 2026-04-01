Aquí está el análisis arquitectónico completo. Primero, un mapa visual de las tres rutas:---

## Análisis Arquitectónico de Rutas para la Métrica de Transferencia

### Ruta A — Basada en CLM (Jeon et al. 2025)

**Formulación matemática general.** Sea $\mathcal{D}_s = \{(x_i, y_i)\}_{i=1}^n$ el conjunto fuente de clasificación con $K$ clases, y $\mathcal{D}_t$ el dominio destino de clustering con restricciones de tamaño $\{(l_k, u_k)\}_{k=1}^K$ sobre la cardinalidad de cada cluster. La métrica CLM-adaptada se define como:

$$\mathcal{T}_A(\mathcal{D}_s, \mathcal{D}_t) = \phi\left(\text{CLM}(\mathcal{D}_s)\right) \cdot \Psi\left(\mathcal{D}_t, \{l_k, u_k\}\right)$$

donde $\phi(\cdot)$ es un funcional de validez global que preserva los axiomas de invariancia (permutación de etiquetas, reescalado, simetría distribucional), y $\Psi(\cdot)$ es un término de penalización por violación de restricciones de tamaño, definido como:

$$\Psi = 1 - \frac{1}{K}\sum_{k=1}^K \mathbf{1}\left[|C_k| \notin [l_k, u_k]\right]$$

La intuición es clara: la métrica hereda el rigor axiomático de CLM, donde la transferibilidad se mide en función de la consistencia distribucional global entre dominios, corregida por el grado en que las restricciones de tamaño son satisfacibles bajo la geometría de $\mathcal{D}_s$.

**Ventajas específicas ante restricciones de tamaño.** El marco axiomático permite demostrar formalmente propiedades de monotonía, lo cual es relevante cuando las restricciones de tamaño se endurecen o relajan. Además, la invariancia a reescalado resulta útil cuando los datasets de clasificación y clustering provienen de poblaciones con distribuciones de clase asimétricas.

**Limitaciones críticas.** El problema fundamental es que CLM fue concebido para medir la validez *interna* de un clustering, no la *transferibilidad* entre tareas. Adaptar sus axiomas al escenario heterogéneo fuente-destino requiere redefinir los axiomas de isomorfismo, lo que debilita la justificación teórica original. Más importante aún: las restricciones de tamaño operan a nivel de asignación individual, mientras que CLM razona sobre distribuciones agregadas. Esta brecha semántica es difícil de resolver sin forzar supuestos que comprometen la generalidad.

---

### Ruta B — Basada en JMDS (Lee et al. 2022)

**Formulación matemática general.** Para cada instancia $x_i \in \mathcal{D}_t$, se define una puntuación de transferibilidad local:

$$\tau_B(x_i, C_k) = \frac{\exp\left(-d(x_i, \mu_k) / \sigma_k\right)}{\sum_{j=1}^K \exp\left(-d(x_i, \mu_j) / \sigma_j\right)}$$

donde $\mu_k$ y $\sigma_k$ son el centroide y la dispersión intrínseca de la clase $k$ estimados desde $\mathcal{D}_s$. La métrica global de transferencia se construye como:

$$\mathcal{T}_B(\mathcal{D}_s, \mathcal{D}_t) = \frac{1}{n}\sum_{i=1}^n \max_k \tau_B(x_i, C_k) \cdot \rho\left(\hat{n}_k, [l_k, u_k]\right)$$

donde $\hat{n}_k = \sum_i \mathbf{1}[\arg\max_j \tau_B(x_i, C_j) = k]$ es el tamaño de cluster inducido, y $\rho(\cdot)$ penaliza la violación de la ventana $[l_k, u_k]$.

La intuición fundamental aquí es geométrica: la métrica cuantifica cuántas instancias del dominio destino encuentran un cluster "natural" en el espacio de representación aprendido por el clasificador fuente, ponderado por si esas asignaciones naturales son compatibles con las restricciones de tamaño.

**Ventajas específicas ante restricciones de tamaño.** JMDS es naturalmente sensible a los desbalances, porque la puntuación $\tau_B(x_i, C_k)$ colapsa para instancias que no tienen vecindad densa en ningún cluster fuente. Esto genera una señal directa: si las restricciones de tamaño obligan a asignar instancias a clusters para los que tienen baja confianza local, la métrica lo detecta. La escalabilidad también es favorable: $\mathcal{T}_B$ es computable mediante una pasada sobre el dataset en $O(nK)$.

**Limitaciones críticas.** La ausencia de axiomas formales hace que la métrica sea difícil de analizar teóricamente en contextos nuevos. Más relevante: JMDS fue diseñado asumiendo que la geometría local del espacio de características es informativa para la tarea destino, lo cual es un supuesto fuerte cuando el dominio de clasificación y el dominio de clustering presentan *covariate shift* severo.

---

### Ruta C — Híbrida CLM-JMDS

**Formulación matemática general.** La ruta híbrida propone una integración convexa con parámetro de mezcla adaptativo:

$$\mathcal{T}_C(\mathcal{D}_s, \mathcal{D}_t) = \alpha(\mathcal{D}_t) \cdot \mathcal{T}_A(\mathcal{D}_s, \mathcal{D}_t) + \left(1 - \alpha(\mathcal{D}_t)\right) \cdot \mathcal{T}_B(\mathcal{D}_s, \mathcal{D}_t)$$

donde $\alpha(\mathcal{D}_t) \in [0,1]$ es una función del dominio destino que regula el balance entre rigor global y sensibilidad local. Una elección natural es:

$$\alpha(\mathcal{D}_t) = \frac{H(\hat{p}_t)}{H_{\max}} \cdot \frac{1}{1 + \gamma \cdot \text{CV}(\{u_k - l_k\})}$$

donde $H(\hat{p}_t)$ es la entropía de la distribución empírica del destino (mayor entropía → más incertidumbre → mayor peso a CLM), y $\text{CV}(\{u_k - l_k\})$ es el coeficiente de variación de los rangos de restricción (mayor heterogeneidad en restricciones → más peso a JMDS).

**Ventajas específicas ante restricciones de tamaño.** Al adaptar $\alpha$ en función de la heterogeneidad de las restricciones, la métrica se vuelve sensible al *tipo* de restricción: restricciones uniformes favorecen el componente global, mientras que restricciones heterogéneas activan el componente local. Esto es conceptualmente atractivo.

**Limitaciones críticas.** La principal es epistémica: la elección de $\alpha$ introduce un meta-parámetro cuya justificación teórica es difícil de solidificar. En una tesis de investigación original, este grado de libertad puede ser percibido por revisores como ad hoc. Adicionalmente, la métrica hereda simultáneamente las debilidades de ambos componentes, sin garantía de que la combinación sea estrictamente mejor. La frontera de Pareto entre ambas métricas base no está caracterizada, lo que dificulta afirmar cuándo $\mathcal{T}_C > \max(\mathcal{T}_A, \mathcal{T}_B)$.

---

### Tabla comparativa bajo los tres criterios

Ahora la comparación estructurada:---

### Conclusión y Recomendación: Ruta B con extensión axiomática selectiva

**La ruta recomendada para una tesis de investigación original es la Ruta B (JMDS), reformulada con un marco teórico propio que incorpore el tratamiento explícito de restricciones de tamaño como ciudadano de primera clase en la definición de la métrica.**

La justificación descansa en tres argumentos articulados:

**Por qué se descarta la Ruta A.** El problema central no es el rigor de CLM en sí, sino que sus axiomas fueron diseñados para medir *cohesión y separación interna* de un clustering, no transferibilidad cruzada. Aplicar esos axiomas al problema de diseño de una métrica de transferencia implica resolver primero un problema de adaptación teórica que es, en sí mismo, una contribución independiente. En lugar de construir sobre una base conceptualmente alineada con el problema, la Ruta A obliga a gastar capital teórico en justificar la adaptación. Ante restricciones de tamaño, el desajuste es todavía mayor: CLM no tiene un mecanismo nativo para razonar sobre cardinalidades de cluster, lo que introduce el término $\Psi$ como un parche externo que fragiliza la coherencia axiomática.

**Por qué se descarta la Ruta C.** La ruta híbrida comete el error clásico de la sobre-ingeniería teórica en etapas tempranas: intenta resolver simultáneamente las limitaciones de dos enfoques mediante un parámetro de mezcla cuya elección no puede justificarse a priori desde primeros principios. En investigación original, la combinación convexa sin una teoría de cuándo $\alpha^* \neq 0, 1$ es equivalente a no haber resuelto el problema de selección de ruta. Los revisores de tesis preguntarán inevitablemente: ¿bajo qué condiciones la métrica híbrida es estrictamente superior a sus componentes? Si esa respuesta no es derivable analíticamente, la contribución se debilita.

**Por qué la Ruta B es el terreno correcto.** JMDS opera en la granularidad natural del problema de clustering con restricciones: la instancia individual. Las restricciones de tamaño ($l_k \leq |C_k| \leq u_k$) son precisamente condiciones sobre asignaciones individuales agregadas, y $\tau_B(x_i, C_k)$ provee exactamente la señal que un algoritmo de asignación restringida necesita: la compatibilidad geométrica de $x_i$ con el prototipo de $C_k$. La extensión natural que constituye la contribución original de la tesis es formalizar la noción de *transferibilidad bajo restricción* como una función de la distribución de confianzas locales proyectada sobre el politopo de restricciones de tamaño:

$$\mathcal{T}^*(\mathcal{D}_s, \mathcal{D}_t, \mathcal{C}) = \mathbb{E}_{x \sim \mathcal{D}_t}\left[\max_{k : \text{asig. factible}} \tau_B(x, C_k)\right]$$

donde "asignación factible" se define respecto al politopo $\mathcal{P} = \{(n_1,\ldots,n_K) : l_k \leq n_k \leq u_k, \sum n_k = n\}$. Esta formulación permite demostrar que la métrica colapsa a JMDS estándar cuando las restricciones son no activas, y que es monótonamente decreciente en el grado de infactibilidad del politopo, lo que constituye un par de axiomas derivables, no postulados.

La estrategia recomendada para la tesis es, entonces, partir de JMDS como base empíricamente validada, enunciar sus limitaciones axiomáticas explícitamente, y demostrar que la extensión propuesta satisface un conjunto mínimo de axiomas de transferencia diseñados desde cero para el problema específico. Esto produce una contribución limpia, justificada y original.