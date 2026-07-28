# causal-ga-dml

**Especificación automatizada en descubrimiento causal mediante el enfoque genético:
sinergia entre Algoritmos Genéticos y Double Machine Learning**

Implementación reproducible del pipeline que sustenta el proyecto de tesis doctoral
y el artículo asociado.

Autor: **Diego Alonso Córdova Ayala** · Asesor: Dr. Jaime Lincovil
Doctorado en Ciencias e Ingeniería Estadística — Universidad Nacional de Ingeniería (UNI), Perú
Taller de Inferencia Causal (DES-304) · Entregable E3

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## 1. Qué resuelve este repositorio

El Modelado Causal Estructural exige un Grafo Acíclico Dirigido (DAG) correctamente
especificado antes de poder estimar cualquier efecto de intervención. Especificarlo a
mano depende del conocimiento de dominio y es frágil: basta confundir un colisionador
o un mediador con un confusor para inducir un sesgo que ningún estimador corrige.

Este repositorio evalúa una alternativa: **descubrir el DAG automáticamente con un
algoritmo genético (AG) y alimentar con él a un estimador Double Machine Learning
(DML)**. La pregunta empírica es concreta: *¿la estructura recuperada por el AG basta
para que el DML entregue un efecto indistinguible del que se obtendría con el DAG
verdadero?*

Se responde sobre la red bayesiana de referencia **ALARM** (37 nodos, 46 aristas), cuyo
DAG conocido actúa como verdad fundamental, y se contrasta el AG contra tres métodos de
referencia: Hill-Climbing con el mismo score, el algoritmo PC y una ablación de orden
aleatorio.

## 2. Resultado principal

Estudio Monte Carlo con **30 réplicas independientes** (media ± desviación estándar):

| Método | F1 esqueleto | Recall esqueleto | F1 dirigido | SHD | Cobertura de confusores | ATE estimado | \|Sesgo\| vs. oráculo | Tiempo |
|---|---|---|---|---|---|---|---|---|
| **AG (este trabajo)** | **0,689 ± 0,020** | **0,822 ± 0,021** | **0,535 ± 0,048** | **42,6 ± 4,0** | **1,000** | 0,319 ± 0,023 | **0,0020** | 9,1 s |
| Hill-Climbing (BIC) | 0,649 ± 0,027 | 0,798 ± 0,026 | 0,449 ± 0,074 | 51,2 ± 6,9 | 0,700 | 0,310 ± 0,030 | 0,0123 | 1,8 s |
| PC (Fisher-z) | 0,675 ± 0,021 | 0,806 ± 0,021 | 0,472 ± 0,026 | 46,9 ± 2,9 | 0,650 | 0,341 ± 0,028 | 0,0257 | 0,7 s |
| Orden aleatorio (ablación) | 0,591 ± 0,029 | 0,785 ± 0,035 | 0,314 ± 0,056 | 67,0 ± 5,7 | 0,567 | 0,327 ± 0,027 | 0,0173 | < 0,1 s |

Referencias: ATE con el DAG verdadero = **0,319 ± 0,023**; ATE sin ajuste alguno =
0,325 ± 0,013.

Tres hallazgos:

1. **El AG es el único método que recupera el 100 % de los confusores verdaderos en
   las 30 réplicas.** Hill-Climbing recupera el 70 %, PC el 65 % y el orden aleatorio
   el 57 %.
2. **Su sesgo respecto del oráculo es un orden de magnitud menor** (0,0020 frente a
   0,0123–0,0257), pese a que ningún método recupera el DAG completo.
3. **El AG nunca contamina el conjunto de ajuste con descendientes del tratamiento.**
   Esa contaminación —que induce sesgo de colisionador— ocurre en el 6,7 % de las
   réplicas con Hill-Climbing, el 23,3 % con PC y el 46,7 % con orden aleatorio.

Los tres refutadores aprueban en el 100 % de las réplicas: causa común aleatoria
0,3148 ± 0,0230 (sin cambio), tratamiento placebo −0,0006 ± 0,0119 (colapsa a cero) y
submuestra al 80 % 0,3165 ± 0,0319 (estable).

La lectura de fondo no es que el AG aprenda mejor el grafo completo, sino que **acierta
en la parte del grafo que la identificación necesita**: la vecindad del tratamiento.

## 3. Instalación

Requiere Python 3.10 o superior.

```bash
git clone https://github.com/<usuario>/causal-ga-dml.git
cd causal-ga-dml

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Con conda:

```bash
conda env create -f environment.yml
conda activate causal-ga-dml
```

## 4. Ejecución

### Corrida única (~30 s)

```bash
python scripts/run_single.py --config configs/default.yaml
```

Imprime las métricas por método y guarda `results/single_seed42.json`.

### Experimento completo del artículo (~12 min)

```bash
make all
# equivale a:
python scripts/run_replicas.py --config configs/multiseed.yaml --reps 30
python scripts/aggregate.py    --config configs/multiseed.yaml
```

Genera `results/multiseed_30.json`, `results/tabla_comparativa.csv` y las cuatro
figuras en `figures/`.

**La ejecución es resumible.** Cada réplica se guarda por separado en
`results/replicas/`; si el proceso se interrumpe, basta volver a lanzarlo para que
continúe donde estaba. Para entornos con límite de tiempo por proceso:

```bash
python scripts/run_replicas.py --reps 30 --budget 600      # lotes de 10 minutos
python scripts/run_replicas.py --reps 30 --offset 0 --stride 2 &   # dos procesos
python scripts/run_replicas.py --reps 30 --offset 1 --stride 2 &   # en paralelo
```

Como cada réplica depende únicamente de su semilla, ni el troceado ni el paralelismo
alteran los resultados.

### Interfaz de línea de comandos

```bash
python -m causal_ga_dml.cli single    --config configs/default.yaml --seed 7
python -m causal_ga_dml.cli multiseed --config configs/multiseed.yaml --reps 30
```

### Pruebas

```bash
make install-dev
make test        # 35 pruebas
make lint
```

## 5. Estructura del repositorio

```
causal-ga-dml/
├── configs/
│   ├── default.yaml           # corrida única, semilla 42
│   ├── multiseed.yaml         # estudio Monte Carlo de 30 réplicas
│   └── quick.yaml             # configuración rápida para pruebas
├── src/causal_ga_dml/
│   ├── config.py              # configuración tipada + registro del entorno
│   ├── seeds.py               # control centralizado de la aleatoriedad
│   ├── data.py                # simulación desde la red ALARM
│   ├── scoring.py             # score gaussiano BIC descomponible
│   ├── ga.py                  # algoritmo genético basado en ordenamientos
│   ├── baselines.py           # Hill-Climbing, PC y ablación de orden aleatorio
│   ├── metrics.py             # SHD, precisión/recall/F1, calidad del ajuste
│   ├── dml.py                 # Double Machine Learning con cross-fitting
│   ├── refuters.py            # refutadores de robustez
│   ├── experiment.py          # orquestación y agregación Monte Carlo
│   ├── plots.py               # figuras del artículo
│   └── cli.py                 # interfaz de línea de comandos
├── scripts/
│   ├── run_single.py          # corrida única
│   ├── run_replicas.py        # réplicas incrementales y resumibles
│   ├── aggregate.py           # agregación, tabla comparativa y figuras
│   └── make_document_tables.py # reconstruye las Tablas 2–6 del documento
├── tests/                     # 35 pruebas unitarias y de reproducibilidad
├── docs/
│   ├── REPRODUCIBILITY.md     # semillas, hardware, versiones, checklist
│   └── METHODS.md             # formalización matemática del pipeline
├── results/                   # salidas (regenerables)
└── figures/                   # figuras (regenerables)
```

## 6. El pipeline en siete etapas

| # | Etapa | Módulo |
|---|---|---|
| 1 | Simulación de 5 000 observaciones desde el DAG verdadero de ALARM | `data.py` |
| 2 | Score gaussiano BIC descomponible por varianza parcial (Schur) | `scoring.py` |
| 3 | AG *order-based*: permutaciones → aciclicidad garantizada; cruce OX, mutación por intercambio, torneo, elitismo | `ga.py` |
| 4 | Métodos de referencia: Hill-Climbing, PC (Fisher-z), orden aleatorio | `baselines.py` |
| 5 | Evaluación estructural: SHD, precisión, recall, F1 y calidad del conjunto de ajuste | `metrics.py` |
| 6 | Estimación del ATE con DML (cross-fitting + bosques aleatorios) | `dml.py` |
| 7 | Refutadores: causa común aleatoria, tratamiento placebo, submuestra al 80 % | `refuters.py` |

## 7. Hiperparámetros

Todos viven en `configs/*.yaml` y se serializan junto a los resultados.

| Bloque | Parámetro | Valor |
|---|---|---|
| Datos | red / n muestras / tratamiento / resultado | ALARM / 5 000 / `CO` / `BP` |
| Score | penalización BIC / regularización | 3,5 / 1e-6 |
| AG | población / generaciones / grado de entrada máximo | 40 / 60 / 4 |
| AG | p. cruce / p. mutación / torneo / elitismo | 0,90 / 0,30 / 3 / 2 |
| DML | particiones / árboles / profundidad / `n_jobs` | 2 / 100 / 6 / 1 |
| PC | α / tamaño máximo del condicionante | 0,01 / 3 |
| Monte Carlo | réplicas / semilla maestra | 30 / 42 |

## 8. Reproducibilidad

- **Semillas.** Una semilla maestra (42) gobierna todo. Las semillas de las réplicas se
  derivan con `numpy.random.SeedSequence`, lo que garantiza flujos independientes y
  reproducibles. Ver `src/causal_ga_dml/seeds.py`.
- **Determinismo.** El AG usa una instancia local `random.Random(seed)` en lugar del
  estado global del módulo `random`; los bosques aleatorios se ejecutan con
  `n_jobs=1` y `random_state` fijo. Las pruebas de `tests/test_reproducibility.py`
  verifican esta propiedad de extremo a extremo.
- **Entorno.** Cada archivo de resultados incluye un bloque `entorno` con el hardware,
  el sistema operativo y las versiones exactas de todas las librerías.
- **Hardware y versiones utilizados.** Ver [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## 9. Salidas generadas

| Archivo | Contenido |
|---|---|
| `results/single_seed42.json` | Corrida de referencia completa, incluidos los DAG |
| `results/multiseed_30.json` | Resumen Monte Carlo + las 30 corridas + entorno |
| `results/tabla_comparativa.csv` | Tabla comparativa lista para el artículo |
| `results/tablas_documento/*.csv` | Tablas 2 a 6 del documento, reconstruidas desde los resultados |
| `figures/fig1_dags.png` | DAG verdadero vs. DAG recuperado por el AG |
| `figures/fig2_convergencia.png` | Convergencia del AG (mejor vs. media poblacional) |
| `figures/fig3_comparacion_metodos.png` | F1 del esqueleto por método (media ± DE) |
| `figures/fig4_distribucion_ate.png` | Distribución del ATE por método vs. oráculo |

## 10. Limitaciones

1. La orientación de aristas dentro de una clase de equivalencia de Markov no es
   identificable a partir de datos puramente observacionales; de ahí la brecha entre el
   F1 del esqueleto (0,689) y el F1 dirigido (0,535).
2. Las variables categóricas de ALARM se codifican ordinalmente y se puntúan con un
   score gaussiano; es una aproximación estándar y escalable, pero un score específico
   para datos discretos podría mejorar la orientación.
3. La validación se hace sobre una red simulada con estructura conocida. Los resultados
   sobre datos simulados no se transfieren automáticamente a datos reales, donde la
   verdad fundamental es incierta.

## 11. Cita

Si utiliza este código, cite el repositorio mediante [`CITATION.cff`](CITATION.cff).

Conjunto de datos de referencia:

> Beinlich, I. A., Suermondt, H. J., Chavez, R. M., & Cooper, G. F. (1989).
> The ALARM monitoring system: A case study with two probabilistic inference techniques
> for belief networks. En *AIME 89* (pp. 247–256). Springer.
> https://doi.org/10.1007/978-3-642-93437-7_28


## 12. Reproducción en Colab Research en Notebook

!rm -rf causal-ga-dml
!git clone https://github.com/diego-555-dmg/causal-ga-dml.git
%cd causal-ga-dml

!pip install -q -r requirements.txt

!python scripts/run_single.py --config configs/default.yaml

!python scripts/run_multiseed.py --reps 30

!python scripts/make_document_tables.py --config configs/multiseed.yaml

from IPython.display import Image, display
for f in ['fig1_dags.png','fig2_convergencia.png','fig3_comparacion_metodos.png','fig4_distribucion_ate.png']:
    display(Image(f'figures/{f}'))


## 13. Licencia

MIT — ver [LICENSE](LICENSE). Uso académico en el marco del curso DES-304, UNI, Perú.





