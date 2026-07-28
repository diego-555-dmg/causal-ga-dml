# Reproducibilidad

Este documento registra todo lo necesario para que un tercero reproduzca exactamente
las cifras reportadas en el artículo y en el proyecto de tesis.

## 1. Cómo reproducir el experimento completo

```bash
pip install -r requirements.txt
make all
```

Tiempo aproximado: **12 minutos** en el hardware descrito abajo. El resultado son
`results/multiseed_30.json`, `results/tabla_comparativa.csv` y las cuatro figuras.

Para verificar únicamente que el entorno funciona (~30 s):

```bash
python scripts/run_single.py --config configs/default.yaml
```

## 2. Semillas

| Nivel | Mecanismo | Valor |
|---|---|---|
| Semilla maestra | `configs/*.yaml` → `seed` | 42 |
| Semillas de réplica | `numpy.random.SeedSequence(42).spawn(30)` | derivadas, deterministas |
| Simulación de datos | `model.simulate(seed=...)` de pgmpy | semilla de la réplica |
| Algoritmo genético | instancia local `random.Random(seed)` | semilla de la réplica |
| Bosques aleatorios | `random_state` de scikit-learn | semilla de la réplica |
| Particiones del cross-fitting | `KFold(random_state=...)` | semilla de la réplica |
| Refutadores | `numpy.random.default_rng(seed)` | semilla de la réplica |

Las 30 semillas efectivamente utilizadas quedan registradas en el campo `semillas` de
`results/multiseed_30.json`.

**Dos decisiones de diseño que garantizan el determinismo:**

1. El AG no usa el estado global del módulo `random`, sino una instancia local
   `random.Random(seed)`. Con el estado global, cualquier llamada externa a `random`
   —incluso de una librería de terceros— desplazaría la secuencia y rompería la
   reproducibilidad.
2. Los bosques aleatorios se ejecutan con `n_jobs=1`. Más allá de eliminar el
   no-determinismo asociado a la planificación de hilos, en este problema resulta
   además más rápido (0,70 s frente a 1,32 s por llamada DML) porque el sobrecoste de
   coordinación supera la ganancia de paralelizar árboles poco profundos.

## 3. Hardware utilizado

| Elemento | Valor |
|---|---|
| Sistema operativo | Linux-6.8.0-124-generic-x86_64-with-glibc2.35 |
| Arquitectura | x86_64 |
| Procesador | x86_64 |
| Núcleos lógicos | 2 |
| Python | 3.10.12 (CPython) |
| Aceleración por GPU | no utilizada |

El pipeline es puramente de CPU y no requiere GPU. Con 2 núcleos lógicos, una réplica
completa tarda unos 22–35 s; el estudio de 30 réplicas, unos 12 minutos.

## 4. Versiones de las librerías

| Librería | Versión |
|---|---|
| `numpy` | 2.2.6 |
| `pandas` | 2.3.3 |
| `networkx` | 3.4.2 |
| `scikit-learn` | 1.7.2 |
| `scipy` | 1.15.3 |
| `pgmpy` | 1.1.2 |
| `matplotlib` | 3.10.9 |
| `pyyaml` | 6.0.3 |

Las versiones están fijadas en `requirements.txt`. `results/multiseed_30.json` incluye
un bloque `entorno` con el registro automático del entorno de la corrida, de modo que
cualquier resultado archivado es rastreable hasta su contexto de ejecución.

## 5. Hiperparámetros

Todos los hiperparámetros se declaran en `configs/*.yaml` y se serializan dentro de
cada archivo de resultados (campo `configuracion`). Ninguna constante relevante está
escrita dentro del código.

| Bloque | Parámetro | Valor |
|---|---|---|
| Datos | red bayesiana | `alarm` (37 nodos, 46 aristas) |
| Datos | observaciones simuladas | 5 000 |
| Datos | tratamiento / resultado | `CO` / `BP` |
| Score | factor de penalización BIC | 3,5 |
| Score | regularización de la covarianza | 1e-6 |
| AG | tamaño de población | 40 |
| AG | número de generaciones | 60 |
| AG | grado de entrada máximo | 4 |
| AG | probabilidad de cruce (OX) | 0,90 |
| AG | probabilidad de mutación (swap) | 0,30 |
| AG | tamaño del torneo | 3 |
| AG | elitismo | 2 |
| DML | particiones del cross-fitting | 2 |
| DML | árboles por bosque | 100 |
| DML | profundidad máxima | 6 |
| DML | `n_jobs` | 1 |
| PC | nivel de significación α | 0,01 |
| PC | tamaño máximo del conjunto condicionante | 3 |
| Hill-Climbing | iteraciones máximas | 200 |
| Monte Carlo | réplicas | 30 |

## 6. Verificación automática

`tests/test_reproducibility.py` comprueba, como parte de la batería de pruebas, que:

- las semillas hijas son deterministas y mutuamente distintas;
- la simulación de datos es idéntica bajo la misma semilla y distinta bajo semillas
  distintas;
- la red ALARM cargada tiene exactamente 37 nodos y 46 aristas;
- el pipeline completo devuelve el mismo ATE, el mismo SHD y el mismo BIC al
  ejecutarse dos veces con la misma semilla.

```bash
make test
```

## 7. Fuentes conocidas de variación

| Fuente | Efecto | Mitigación |
|---|---|---|
| Versión de pgmpy | El orden interno de los nodos puede cambiar entre versiones | Las columnas se reordenan alfabéticamente en `data.py` |
| Versión de scikit-learn | Cambios en el algoritmo del bosque alteran los residuos | Versión fijada en `requirements.txt` |
| Aritmética BLAS/LAPACK | Diferencias en el último dígito significativo entre plataformas | Sin impacto en las conclusiones; el score se regulariza con un término ridge |
| Paralelismo | Reducciones no deterministas | `n_jobs=1` en el DML |

## 8. Diferencias respecto del prototipo del entregable E2

El código fue reestructurado para este entregable. Tres cambios afectan a los números
publicados y se documentan por transparencia:

1. **Selección por torneo corregida.** El prototipo devolvía `poblacion[min(aspirantes)]`,
   lo que dependía de que la población estuviera ordenada por aptitud y sesgaba la
   presión selectiva. Ahora el torneo compara aptitudes explícitamente.
2. **Aleatoriedad encapsulada.** El prototipo mezclaba una instancia local con el estado
   global de `random`, de modo que no era estrictamente reproducible. Ahora toda la
   aleatoriedad del AG proviene de una única instancia local.
3. **Exclusión del resultado del conjunto de ajuste.** Si un método de descubrimiento
   orienta mal la arista tratamiento–resultado, el resultado entra entre los "padres"
   del tratamiento y condicionar sobre él anula la varianza residual del DML. La
   salvaguarda replica lo que haría cualquier analista; la contaminación estructural se
   reporta aparte como métrica (`contamina_descendientes`).

Estos cambios mejoran las métricas del AG respecto del prototipo (F1 del esqueleto de
0,67 a 0,70 y SHD de 48 a 41 con la semilla 42) y hacen que las cifras sean auditables.
