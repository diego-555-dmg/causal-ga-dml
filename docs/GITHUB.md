# Publicación en GitHub

Instrucciones para subir este repositorio y entregarlo como enlace.

## 1. Crear el repositorio remoto

En GitHub: **New repository** → nombre `causal-ga-dml` → visibilidad *Public* (o
*Private* con el evaluador añadido como colaborador) → **no** marcar «Add a README»,
«Add .gitignore» ni «Choose a license»: el repositorio ya los incluye.

## 2. Subir el contenido

Desde la carpeta `causal-ga-dml/`:

```bash
git init
git add .
git commit -m "Entregable E3: pipeline reproducible AG + Double Machine Learning"
git branch -M main
git remote add origin https://github.com/<usuario>/causal-ga-dml.git
git push -u origin main
```

Si es la primera vez que se usa git en la máquina:

```bash
git config --global user.name  "Diego Alonso Córdova Ayala"
git config --global user.email "diego.cordova@dmg-pe.com"
```

GitHub ya no acepta contraseña por HTTPS: use un **personal access token**
(*Settings → Developer settings → Personal access tokens → Fine-grained tokens*, con
permiso `Contents: Read and write`) y péguelo cuando pida la contraseña.

## 3. Verificar que el repositorio queda limpio

`.gitignore` excluye las salidas regenerables (`results/*.json`, `figures/*.png`,
`__pycache__/`). Compruebe antes del commit:

```bash
git status --short
```

Si desea **incluir** los resultados y figuras del artículo para que el evaluador los vea
sin ejecutar nada:

```bash
git add -f results/multiseed_30.json results/single_seed42.json \
          results/tabla_comparativa.csv figures/*.png
```

## 4. Etiquetar la versión entregada

```bash
git tag -a v1.0.0 -m "Entregable E3 - versión evaluada"
git push origin v1.0.0
```

## 5. Comprobación final antes de entregar

- [ ] `make test` pasa (35 pruebas).
- [ ] `python scripts/run_single.py` corre de principio a fin en una máquina limpia.
- [ ] El README muestra correctamente las tablas en GitHub.
- [ ] `LICENSE` y `CITATION.cff` están presentes.
- [ ] Los números del README coinciden con `results/tabla_comparativa.csv`.
- [ ] El enlace del repositorio aparece en el artículo y en el proyecto de tesis.

## 6. Opcional: integración continua

Guarde lo siguiente como `.github/workflows/tests.yml` para que GitHub ejecute las
pruebas en cada `push`:

```yaml
name: tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.10"
      - run: pip install -r requirements-dev.txt
      - run: python -m pytest -q
```
