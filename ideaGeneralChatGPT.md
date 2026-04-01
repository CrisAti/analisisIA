# Análisis arquitectónico de rutas de diseño para una métrica de transferencia

A continuación presento un análisis arquitectónico de las tres rutas posibles para tu métrica de transferencia. Tomo como base dos hechos del estado del arte:  
CLM plantea que, para comparar datasets etiquetados como benchmarks de clustering, la medida debe reflejar el **grado de correspondencia entre etiquetas y estructura de clúster** y, además, ser **invariante a propiedades no estructurales** del dataset cuando se compara entre datasets. ([arxiv.org](https://arxiv.org/abs/2503.01097))  
JMDS, en cambio, fue diseñado como una **confianza local por muestra**, combinando una señal geométrica del dato (LPG) con una señal del modelo (MPPL), y su score final es el producto de ambas. ([proceedings.mlr.press](https://proceedings.mlr.press/v162/lee22c/lee22c.pdf))

---

# 1. Planteamiento general del problema

Sea un dataset etiquetado

```math
\mathcal D = \{(x_i,y_i)\}_{i=1}^n,\qquad x_i\in\mathbb R^d,\; y_i\in\{1,\dots,K\}.
```

Sea \(n_k = |\{i:y_i=k\}|\) y la proporción empírica

```math
\pi_k = \frac{n_k}{n},\qquad \pi=(\pi_1,\dots,\pi_K)\in\Delta^{K-1}.
```

El clustering destino no es libre: debe cumplir restricciones de tamaño. La forma más general es una **región factible**:

```math
\Pi^\star \subseteq \Delta^{K-1}.
```

Esta región puede representar:

- proporciones objetivo no uniformes,
- cotas inferiores y superiores,
- capacidades,
- desbalance severo,
- o cualquier restricción convexa/poliedral razonable.

La métrica que buscamos debe responder:

> **¿Qué tan apto es este dataset etiquetado para ser reutilizado como problema de clustering bajo restricciones de tamaño \(\Pi^\star\)?**

La idea clave es que esta aptitud no depende solo de “qué tan separadas” están las clases, sino también de **cuánto habría que deformar** esa partición natural para volverla compatible con \(\Pi^\star\).

---

# 2. Ruta A — Arquitectura basada en CLM

## 2.1 Intuición

La Ruta A interpreta el problema desde una óptica **global**:

1. Primero mide si las etiquetas del dataset son una buena partición “tipo clustering”.
2. Luego mide cuánto entran en conflicto esas etiquetas con la región factible \(\Pi^\star\).

Es la ruta más cercana al espíritu de CLM: una métrica bien diseñada debe comparar datasets sin contaminarse por tamaño muestral, dimensionalidad u otras propiedades no estructurales. ([arxiv.org](https://arxiv.org/abs/2503.01097))

---

## 2.2 Formulación propuesta

## (a) Score estructural global

Define un funcional global \(G(\mathcal D)\in[0,1]\) que mida qué tan bien la partición inducida por \(Y\) coincide con una estructura clusterizable. Puede venir de una versión ajustada de un IVM:

```math
G(\mathcal D) := \widetilde{\mathrm{IVM}}(X,Y).
```

Aquí \(\widetilde{\mathrm{IVM}}\) representa una versión estandarizada/ajustada al estilo CLM para comparación entre datasets. La arquitectura no depende de un IVM particular: puede ser una forma ajustada de Silhouette, Dunn, Davies–Bouldin, etc., siempre que respete la filosofía de invariancia de CLM. ([arxiv.org](https://arxiv.org/abs/2503.01097))

---

## (b) Distancia de factibilidad de tamaños

Mide el conflicto entre la distribución de clases observada y la región factible:

```math
F(\pi,\Pi^\star) := \min_{q\in\Pi^\star} D(\pi,q),
```

donde \(D\) puede ser una divergencia sin hiperparámetros, por ejemplo:

- distancia \(L_1\),
- total variation,
- Wasserstein-1 en el simplex si hay estructura entre clases,
- o Bregman simple si el problema lo justifica.

Para mantener interpretabilidad y no introducir calibración, una opción robusta es:

```math
F(\pi,\Pi^\star)=\min_{q\in\Pi^\star}\frac{1}{2}\|\pi-q\|_1.
```

Este término tiene una interpretación directa: es la **fracción mínima de masa que debe reubicarse** para cumplir la restricción de tamaño.

---

## (c) Métrica Ruta A

Una forma natural es:

```math
T_A(\mathcal D,\Pi^\star)=G(\mathcal D)\,\bigl(1-F(\pi,\Pi^\star)\bigr).
```

También puede escribirse como media armónica si quieres castigar más fuerte el peor término:

```math
T_A^{(H)} = \frac{2\,G(1-F)}{G+(1-F)}.
```

---

## 2.3 Lectura conceptual

- \(G\) dice: “las etiquetas se parecen a clústeres reales”.
- \(1-F\) dice: “esas clases son compatibles con las restricciones de tamaño”.

Entonces \(T_A\) es alto solo cuando ambas cosas ocurren a la vez.

---

## 2.4 Ventajas frente a restricciones de tamaño

1. **Muy limpio teóricamente.**  
   La parte de CLM aporta un marco fuertemente principista. ([arxiv.org](https://arxiv.org/abs/2503.01097))

2. **Muy interpretable.**  
   Se separa en dos factores visibles: estructura y factibilidad.

3. **Muy escalable.**  
   Si \(G\) se computa con un IVM ajustado eficiente y \(F\) es una proyección al simplex restringido, el costo es bajo.

---

## 2.5 Limitaciones

1. **No ve dónde está la masa “movible”.**  
   Dos datasets pueden tener el mismo \(\pi\) y el mismo \(G\), pero en uno las muestras que habría que reasignar están en fronteras ambiguas y en otro están en núcleos muy compactos. Ruta A no distingue eso.

2. **La penalización de tamaño es demasiado gruesa.**  
   Castiga “cuánta masa” se mueve, no “qué tan doloroso estructuralmente” es moverla.

3. **Puede sobrerigidizar el problema.**  
   En clustering con capacidades, no toda desviación de \(\pi\) implica igual dificultad real.

---

# 3. Ruta B — Arquitectura basada en JMDS

## 3.1 Intuición

La Ruta B mira el problema desde abajo: desde la **muestra individual**.

JMDS combina una señal de estructura local del dato con una señal del modelo; en el paper original esto se implementa como producto entre LPG y MPPL. ([proceedings.mlr.press](https://proceedings.mlr.press/v162/lee22c/lee22c.pdf))  
La idea aquí es reinterpretar esa lógica para tu tesis:

- **confianza local** = qué tan firmemente una muestra pertenece a su clase;
- si hay conflicto con \(\Pi^\star\), conviene “expulsar” primero muestras de baja confianza.

Esta ruta es particularmente atractiva para restricciones de tamaño porque el problema real no es solo de proporciones globales, sino de **qué muestras concretas deberían ser desplazadas**.

---

## 3.2 Formulación propuesta

## (a) Confianza local por muestra

Para cada muestra \(x_i\), define:

```math
u_i = s_i \cdot m_i,\qquad u_i\in[0,1].
```

donde:

- \(s_i\) = confianza geométrica local, análoga al LPG,
- \(m_i\) = confianza del modelo respecto a la etiqueta, análoga al MPPL. ([proceedings.mlr.press](https://proceedings.mlr.press/v162/lee22c/lee22c.pdf))

### Parte geométrica local \(s_i\)

Puedes definirla sin entrenar un modelo complejo, por ejemplo mediante densidades locales por clase:

```math
s_i
=
\frac{\rho_{y_i}(x_i)-\max_{b\neq y_i}\rho_b(x_i)}
{\rho_{y_i}(x_i)+\varepsilon},
```

recortada a \([0,1]\), donde \(\rho_k\) es una densidad local o proxy de densidad para la clase \(k\) en el espacio embebido.

Otra versión más escalable es con vecinos:

```math
s_i=\frac{1}{k}\sum_{j\in\mathcal N_k(i)} \mathbf 1(y_j=y_i),
```

ajustada por margen respecto a la segunda clase más frecuente en el vecindario.

### Parte de modelo \(m_i\)

Entrena un clasificador liviano \(f\) sobre \((X,Y)\) y usa probabilidad leave-one-out:

```math
m_i = p_f(y_i\mid x_i).
```

Así, \(u_i\) es alto cuando la muestra es localmente coherente y además el modelo la considera prototípica.

---

## (b) Costo mínimo de ajuste a restricciones de tamaño

Para cada clase \(k\), si elegimos una proporción factible \(q_k\in\Pi^\star\), el exceso de masa es

```math
e_k(q)= (\pi_k-q_k)_+.
```

Ese exceso debe salir de la clase \(k\).  
La idea JMDS es: saquemos primero las muestras menos confiables.

Sea \(u_{k,(1)}\le \cdots \le u_{k,(n_k)}\) el orden ascendente de las confianzas dentro de la clase \(k\). Entonces el costo mínimo de expulsar masa de la clase \(k\) es

```math
C_k(q)=\frac{1}{n}\sum_{r=1}^{\lfloor e_k(q)n\rfloor} u_{k,(r)}.
```

Y el costo total:

```math
C_{\text{loc}}(q)=\sum_{k=1}^K C_k(q).
```

Finalmente,

```math
C_{\text{loc}}^\star = \min_{q\in\Pi^\star} C_{\text{loc}}(q).
```

---

## (c) Métrica Ruta B

```math
T_B(\mathcal D,\Pi^\star)
=
1 - C_{\text{loc}}^\star.
```

Opcionalmente, si deseas normalización por confianza total disponible:

```math
T_B^{(\text{norm})}
=
1-\frac{C_{\text{loc}}^\star}{\frac{1}{n}\sum_{i=1}^n u_i+\varepsilon}.
```

---

## 3.3 Lectura conceptual

- Si una clase está sobrerrepresentada respecto de \(\Pi^\star\), la métrica pregunta:
  **¿puedo ajustar esa clase removiendo solo puntos ambiguos o periféricos?**
- Si sí, el score sigue alto.
- Si para cumplir capacidad debo romper núcleos muy confiables, el score cae fuerte.

---

## 3.4 Ventajas frente a restricciones de tamaño

1. **Modela exactamente el conflicto relevante.**  
   El costo no depende solo de proporciones, sino de la “movilidad estructural” de las muestras.

2. **Muy adecuada para desbalance y capacidades.**  
   Porque distingue entre exceso de masa rígida y exceso de masa blanda.

3. **Escalable si se implementa bien.**  
   Con kNN aproximado, centroides, o embeddings precomputados, puede ser muy eficiente.

---

## 3.5 Limitaciones

1. **Pierde comparabilidad global.**  
   JMDS fue pensado como score por muestra, no como axiomatización across-datasets. ([proceedings.mlr.press](https://proceedings.mlr.press/v162/lee22c/lee22c.pdf))

2. **Más sensible a decisiones de implementación.**  
   Vecindarios, embeddings, clasificador auxiliar o estimador de densidad afectan el resultado.

3. **Puede ser menos interpretable a nivel tesis si no se formaliza muy bien.**  
   Sin un andamiaje axiomático, el aporte puede parecer más heurístico que fundacional.

---

# 4. Ruta C — Arquitectura híbrida CLM–JMDS

## 4.1 Intuición

La Ruta C separa el problema en dos niveles complementarios:

- **Nivel global:**  
  ¿Las etiquetas definen una partición con legitimidad de clustering?  
  Esto lo hereda de CLM.

- **Nivel local de factibilidad:**  
  Si además impongo \(\Pi^\star\), ¿puedo satisfacerla moviendo solo muestras débiles o ambiguas?  
  Esto lo hereda de JMDS.

Esta arquitectura refleja mejor la naturaleza real del problema:  
**size-constrained clustering no destruye completamente la estructura; la deforma localmente.**

---

## 4.2 Formulación propuesta

Usamos:

- \(G(\mathcal D)\in[0,1]\): score estructural global tipo CLM,
- \(C_{\text{loc}}^\star\in[0,1]\): costo mínimo local tipo JMDS para volver factible la partición bajo \(\Pi^\star\).

Define el factor local de preservación:

```math
L(\mathcal D,\Pi^\star)=1-C_{\text{loc}}^\star.
```

Entonces la métrica híbrida natural es

```math
T_C(\mathcal D,\Pi^\star)=\sqrt{\,G(\mathcal D)\,L(\mathcal D,\Pi^\star)\,}.
```

La media geométrica tiene una ventaja importante: **no introduce hiperparámetros** y exige simultáneamente buen comportamiento global y local.

Otra variante más estricta:

```math
T_C^{(H)}(\mathcal D,\Pi^\star)
=
\frac{2\,G\,L}{G+L}.
```

---

## 4.3 Versión aún más formal

Si quieres una formulación de tesis más fuerte, puedes escribir la arquitectura como una composición funcional:

```math
T_C = \Phi\bigl(\underbrace{\mathcal S_{\text{global}}(X,Y)}_{\text{CLM}},
\underbrace{\mathcal S_{\text{local-feasible}}(X,Y,\Pi^\star)}_{\text{JMDS}}\bigr),
```

donde

```math
\mathcal S_{\text{global}}(X,Y)=\widetilde{\mathrm{IVM}}(X,Y),
```

```math
\mathcal S_{\text{local-feasible}}(X,Y,\Pi^\star)
=
1-
\min_{q\in\Pi^\star}
\sum_{k=1}^K
\frac{1}{n}\sum_{r=1}^{\lfloor (\pi_k-q_k)_+n\rfloor} u_{k,(r)},
```

y \(\Phi(a,b)=\sqrt{ab}\) o media armónica.

---

## 4.4 Intuición fuerte de tesis

La métrica híbrida responde dos preguntas distintas:

1. **¿Existe estructura tipo clustering?**  
2. **¿Esa estructura sobrevive cuando impones tamaños no libres?**

Eso hace que no sea una simple extensión de CLM ni una simple adaptación de JMDS, sino una métrica nueva con una motivación clara.

---

## 4.5 Ventajas frente a restricciones de tamaño

1. **Captura el conflicto real del problema.**  
   No solo ve si la partición natural es buena, sino si es **compatible de forma suave** con restricciones de capacidad.

2. **Mantiene rigor sin perder sensibilidad local.**  
   Esa combinación es exactamente lo que tu tesis necesita.

3. **Evita depender de tuning fino.**  
   Si eliges media geométrica/armónica y cuantiles empíricos, no necesitas pesos aprendidos.

4. **Escala razonablemente bien.**  
   La parte global puede ser rápida, y la local puede reducirse a ordenamientos por clase más proyección a \(\Pi^\star\).

---

## 4.6 Limitaciones

1. **Más compleja de demostrar.**  
   Necesitarás dos bloques teóricos: invariancia global y optimalidad local del ajuste.

2. **Más trabajo experimental.**  
   Debes demostrar que el componente híbrido efectivamente supera a versiones solo-globales y solo-locales.

Pero, para una tesis, esto no es una desventaja real: es precisamente donde aparece la originalidad.

---

# 5. Comparación objetiva

| Criterio | Ruta A (CLM) | Ruta B (JMDS) | Ruta C (Híbrida) |
|---|---|---|---|
| **Rigor matemático** | **Alto**: muy alineada con axiomas y comparación across-datasets. ([arxiv.org](https://arxiv.org/abs/2503.01097)) | Medio: fuerte localmente, pero menos fundacional como métrica global. ([proceedings.mlr.press](https://proceedings.mlr.press/v162/lee22c/lee22c.pdf)) | **Muy alto**: combina base axiomática con costo local formalizado |
| **Escalabilidad algorítmica** | **Alta**: IVM ajustado + proyección a \(\Pi^\star\) | Alta–media: depende de kNN/densidades/modelo auxiliar | **Alta–media**: algo más costosa, pero todavía viable para gran escala |
| **Facilidad de interpretación** | **Muy alta**: estructura × factibilidad | Media: más técnica, muestra a muestra | Alta: requiere explicar dos niveles, pero queda muy convincente |
| **Sensibilidad al problema de tamaños** | Media: ve masa global, no qué puntos mover | **Muy alta**: ve exactamente la “movilidad” local | **Muy alta**: ve masa global + movilidad local |
| **Potencial de originalidad de tesis** | Medio | Medio–alto | **Muy alto** |

---

# 6. Conclusión: ruta recomendada

## Selección final: **Ruta C — Híbrida CLM–JMDS**

Es la ruta más robusta para una tesis original.

### Por qué la elijo

Porque tu problema no es simplemente medir clusterabilidad ni simplemente medir confianza local. Tu problema es más específico:

> evaluar si una partición etiquetada sigue siendo una buena base para clustering cuando el clustering destino está sometido a restricciones de tamaño.

Ese problema tiene **dos escalas inseparables**:

- una escala **global**, donde necesitas que la medida sea comparable y conceptualmente sólida;
- una escala **local**, donde necesitas saber si cumplir las restricciones exige romper estructura fuerte o solo reorganizar frontera ambigua.

La Ruta C es la única que modela ambas.

---

## Por qué descarto Ruta A

No la descarto por débil, sino por **insuficiente**.

Es excelente si el problema fuera solo:
“¿el dataset tiene buen CLM?”

Pero en tu tesis eso no basta.  
Con restricciones de tamaño, lo decisivo no es solo la calidad global de la partición, sino el **costo estructural de volverla factible**. Ruta A no ve ese costo a nivel de muestra; por eso puede subestimar datasets muy transferibles o sobrevalorar otros rígidos.

---

## Por qué descarto Ruta B

Tampoco la descarto por mala, sino por **incompleta como tesis fundacional**.

Es potente para modelar el ajuste local bajo restricciones, pero por sí sola no te da un marco tan limpio de comparación entre datasets. Y en una tesis cuyo objeto es una **métrica de transferencia**, ese vacío teórico pesa bastante. Sin el bloque tipo CLM, la propuesta puede verse demasiado dependiente de decisiones de implementación.

---

# 7. Recomendación de formulación final para tu tesis

Si tuviera que dejarte una forma canónica, sería esta:

```math
\boxed{
T(\mathcal D,\Pi^\star)
=
\sqrt{
\widetilde{\mathrm{IVM}}(X,Y)
\left(
1-
\min_{q\in\Pi^\star}
\sum_{k=1}^K
\frac{1}{n}
\sum_{r=1}^{\lfloor (\pi_k-q_k)_+n\rfloor}
u_{k,(r)}
\right)
}
}
```

donde:

- \(\widetilde{\mathrm{IVM}}(X,Y)\) mide legitimidad global de las etiquetas como clústeres,
- \(u_i=s_i m_i\) mide cuán “movible” es cada muestra sin destruir estructura,
- el mínimo sobre \(q\in\Pi^\star\) busca la forma más barata de satisfacer la restricción.

Esa fórmula ya contiene una tesis completa:
**estructura global válida + costo local mínimo de factibilidad**.

Si quieres, en el siguiente paso te la convierto en una **propuesta formal de tesis** con axiomas, definición, propiedades esperadas y plan experimental.
