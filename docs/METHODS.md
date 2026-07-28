# Formalización del pipeline

Documento de referencia matemática. Complementa al README, que describe *cómo* ejecutar
el experimento; aquí se justifica *por qué* cada pieza es la adecuada.

## 1. Marco causal estructural

Un modelo causal estructural es una tupla $M = \langle U, V, F, P(U)\rangle$ donde $V$
son las variables endógenas, $U$ las exógenas y cada $V_i \in V$ se genera como
$V_i = f_i(\mathrm{Pa}_i, U_i)$. El grafo inducido, con una arista $V_j \to V_i$ por cada
$V_j \in \mathrm{Pa}_i$, es un DAG $G$.

El efecto de una intervención $do(T = t)$ sobre $Y$ es identificable a partir de datos
observacionales si existe un conjunto $Z$ que satisfaga el **criterio de la puerta
trasera** respecto de $(T, Y)$:

1. ningún nodo de $Z$ es descendiente de $T$;
2. $Z$ bloquea todo camino entre $T$ e $Y$ que empiece con una flecha hacia $T$.

Entonces
$$P(y \mid do(t)) = \sum_z P(y \mid t, z)\,P(z).$$

**Resultado que explota este trabajo.** El conjunto de padres $\mathrm{Pa}(T)$ satisface
siempre el criterio de la puerta trasera en un DAG causal. Por tanto, *descubrir la
estructura equivale a descubrir un conjunto de ajuste válido*, y la calidad del
descubrimiento puede evaluarse en la métrica que realmente importa: si $\mathrm{Pa}(T)$
estimado contiene los confusores verdaderos y excluye a los descendientes de $T$.

Esto explica el hallazgo central del estudio: el AG no gana porque recupere mejor el
grafo completo —su F1 dirigido es 0,535—, sino porque acierta sistemáticamente en la
vecindad de $T$, que es la única región del grafo de la que depende la identificación.

## 2. Score gaussiano BIC descomponible

Para un nodo $X_i$ con padres $\mathrm{Pa}_i$:

$$\mathrm{score}(X_i \mid \mathrm{Pa}_i) = -\frac{n}{2}\log \hat{\sigma}^2_{i \mid \mathrm{Pa}_i} - \frac{\lambda}{2}\,k\,\log n,
\qquad k = |\mathrm{Pa}_i| + 1$$

La varianza residual se obtiene por complemento de Schur sobre la matriz de covarianza
$S$:

$$\hat{\sigma}^2_{i \mid \mathrm{Pa}_i} = S_{ii} - S_{i,\mathrm{Pa}}\,S_{\mathrm{Pa},\mathrm{Pa}}^{-1}\,S_{\mathrm{Pa},i}$$

Dos propiedades hacen viable el AG:

- **Descomponibilidad.** El score global es la suma de los locales, de modo que cambiar
  los padres de un nodo no obliga a reevaluar el resto del grafo.
- **Costo independiente de $n$.** Tras precomputar $S$ una sola vez, cada evaluación
  cuesta $O(|\mathrm{Pa}|^3)$ y no $O(n\,p)$. En la corrida de referencia esto permite
  26 516 evaluaciones de score en 9 segundos.

Con $\lambda = 1$ se recupera el BIC clásico. Se usa $\lambda = 3{,}5$: la penalización
más fuerte reduce los falsos positivos estructurales, que son el error costoso cuando
el objetivo es un conjunto de ajuste limpio.

## 3. Algoritmo genético basado en ordenamientos

**Representación.** Cada individuo es una permutación $\pi$ de los $p$ nodos. Dado
$\pi$, los padres de cada nodo se eligen por avidez entre sus predecesores. La
aciclicidad se satisface por construcción.

Esto reduce el espacio de búsqueda de los $O(3^{p^2})$ grafos dirigidos a $p!$
ordenamientos, y elimina el costo de reparar u ordenar grafos inválidos, que es el
cuello de botella de las representaciones matriciales. La contrapartida —el resultado
depende del orden— es precisamente lo que el AG optimiza, en lugar de fijarlo
arbitrariamente como hacen los métodos que dependen del orden de lectura de las
variables.

**Operadores.**

| Operador | Elección | Razón |
|---|---|---|
| Cruce | Order Crossover (OX), $p = 0{,}90$ | Preserva la validez de la permutación sin reparación |
| Mutación | Intercambio de dos posiciones, $p = 0{,}30$ | Perturbación local mínima que conserva la permutación |
| Selección | Torneo de tamaño 3 | Presión selectiva moderada, controlable e insensible a la escala del score |
| Elitismo | 2 individuos | Garantiza monotonía del mejor individuo entre generaciones |

La ablación de *orden aleatorio* del repositorio aísla la contribución de esta
maquinaria: comparte representación y decodificación con el AG, pero carece de
selección, cruce y mutación. Su F1 de esqueleto cae a 0,591 y su cobertura de
confusores a 0,567, frente a 0,689 y 1,000 del AG.

## 4. Double Machine Learning

Modelo parcialmente lineal:

$$Y = \theta_0 T + g_0(Z) + U, \qquad T = m_0(Z) + V$$

con $\mathbb{E}[U \mid Z, T] = 0$ y $\mathbb{E}[V \mid Z] = 0$. El estimador ortogonal de
Neyman con cross-fitting es

$$\hat{\theta} = \frac{\widehat{\mathrm{Cov}}(\tilde{Y}, \tilde{T})}{\widehat{\mathrm{Var}}(\tilde{T})},
\qquad \tilde{Y} = Y - \hat{g}(Z), \quad \tilde{T} = T - \hat{m}(Z)$$

donde $\hat{g}$ y $\hat{m}$ se ajustan fuera de muestra. La ortogonalización elimina el
sesgo de regularización de primer orden y el cross-fitting el de sobreajuste, lo que
entrega convergencia $\sqrt{n}$ aun con aprendices no paramétricos.

El error estándar se obtiene de la varianza sándwich de la función de momentos
$\psi = (\tilde{Y} - \theta \tilde{T})\,\tilde{T}$.

**Punto clave.** El DML es *estructuralmente ciego*: garantiza insesgadez condicionada a
recibir un $Z$ válido, pero no tiene forma de verificar esa validez. Ahí es donde entra
el descubrimiento estructural, y por eso la métrica relevante para comparar métodos de
descubrimiento no es el SHD global sino la calidad del conjunto de ajuste que producen.

## 5. Métricas

| Métrica | Definición | Qué mide |
|---|---|---|
| SHD | aristas ausentes + sobrantes + invertidas | Distancia estructural global |
| Precisión/recall/F1 dirigidos | sobre aristas orientadas | Recuperación con orientación |
| Precisión/recall/F1 de esqueleto | sobre adyacencias no orientadas | Recuperación de adyacencias |
| Cobertura de confusores | $\lvert \mathrm{Pa}(T)_{\text{est}} \cap \mathrm{Pa}(T)_{\text{true}}\rvert / \lvert \mathrm{Pa}(T)_{\text{true}}\rvert$ | Suficiencia del conjunto de ajuste |
| Contaminación por descendientes | $\mathbb{1}[\mathrm{Pa}(T)_{\text{est}} \cap \mathrm{De}(T) \neq \emptyset]$ | Riesgo de sesgo de colisionador |
| Sesgo vs. oráculo | $\lvert \hat{\theta}_{\text{método}} - \hat{\theta}_{\text{DAG verdadero}}\rvert$ | Consecuencia estimacional del error estructural |

La separación entre métricas dirigidas y de esqueleto es necesaria: los scores basados
en verosimilitud son equivalentes dentro de una clase de equivalencia de Markov, de modo
que ninguna cantidad de datos observacionales identifica la orientación de ciertas
aristas. Reportar solo el F1 dirigido penalizaría al método por un límite teórico, no
por un fallo del método.

## 6. Refutadores

| Refutador | Manipulación | Criterio de aprobación | Resultado (30 réplicas) |
|---|---|---|---|
| Causa común aleatoria | Añade un confusor $N(0,1)$ irrelevante | Desvío relativo ≤ 15 % | 0,3148 ± 0,0230 ✓ |
| Tratamiento placebo | Permuta $T$ | $\lvert\hat{\theta}\rvert \le 0{,}15\,\lvert\hat{\theta}_{\text{base}}\rvert$ | −0,0006 ± 0,0119 ✓ |
| Submuestra al 80 % | Remuestrea sin reposición | Desvío relativo ≤ 15 % | 0,3165 ± 0,0319 ✓ |

Un refutador no confirma la hipótesis causal: la somete a una prueba que debería fallar
si el efecto fuese un artefacto. Los tres aprueban en el 100 % de las réplicas.
