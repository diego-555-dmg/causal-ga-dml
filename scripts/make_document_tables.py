#!/usr/bin/env python3
"""Reconstruye TODAS las tablas del proyecto de tesis y del artículo a partir
de los resultados del experimento, y las guarda como archivos CSV.

Mientras que `aggregate.py` produce un único CSV comparativo, este script emite
cada tabla del documento por separado (Tabla 3, 4, 5 y 6), de modo que al correr
el código en Colab se obtenga exactamente el mismo material que figura en los
`.docx`. Las Tablas 1 y 2 son conceptuales/de configuración y se incluyen como
texto de referencia.

    python scripts/make_document_tables.py --config configs/multiseed.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from causal_ga_dml.config import load_config  # noqa: E402

NOMBRES = {"AG": "Algoritmo genético", "Hill-Climbing": "Hill-Climbing (BIC)",
           "PC": "PC (Fisher-z)", "Orden aleatorio": "Orden aleatorio (ablación)"}


def ms(bloque: dict, decimales: int = 3) -> str:
    """Formatea 'media ± desviación' con coma decimal, al estilo del documento."""
    m = f"{bloque['media']:.{decimales}f}".replace(".", ",")
    d = f"{bloque['de']:.{decimales}f}".replace(".", ",")
    return f"{m} ± {d}"


def coma(x: float, decimales: int = 3) -> str:
    return f"{x:.{decimales}f}".replace(".", ",")


def tabla3(resumen: dict) -> pd.DataFrame:
    filas = []
    for m, bloque in resumen["por_metodo"].items():
        e = bloque["estructura"]
        filas.append({
            "Método": NOMBRES[m],
            "F1 esqueleto": ms(e["f1_esqueleto"]),
            "Recall esqueleto": ms(e["recall_esqueleto"]),
            "Precisión esqueleto": ms(e["precision_esqueleto"]),
            "F1 dirigido": ms(e["f1_dirigido"]),
            "SHD": ms(e["SHD"], 1),
        })
    return pd.DataFrame(filas)


def tabla4(resumen: dict) -> pd.DataFrame:
    filas = []
    for m, bloque in resumen["por_metodo"].items():
        filas.append({
            "Método": NOMBRES[m],
            "Cobertura de confusores": coma(bloque["cobertura_confusores"]["media"]),
            "Contaminación por descendientes":
                coma(bloque["tasa_contaminacion_descendientes"] * 100, 1) + " %",
            "|Sesgo| vs. oráculo": coma(bloque["sesgo_absoluto_vs_oraculo"]["media"], 4),
        })
    return pd.DataFrame(filas)


def tabla5(resumen: dict) -> pd.DataFrame:
    pm, ref, rob = resumen["por_metodo"], resumen["referencias"], resumen["robustez"]
    filas = [
        ("Sin ajuste alguno", ms(ref["ATE_sin_ajuste"]), "Referencia sesgada"),
        ("Ajuste con el DAG verdadero (oráculo)", ms(ref["ATE_dag_verdadero"]), "Referencia insesgada"),
        ("Ajuste con el DAG del algoritmo genético", ms(pm["AG"]["ATE"]), "Coincide con el oráculo"),
        ("Ajuste con el DAG del Hill-Climbing", ms(pm["Hill-Climbing"]["ATE"]), "Sesgo moderado"),
        ("Ajuste con el DAG del PC", ms(pm["PC"]["ATE"]), "Sobreestimación"),
        ("Refutador: causa común aleatoria", ms(rob["causa_comun_aleatoria"]), "Estable"),
        ("Refutador: tratamiento placebo", ms(rob["tratamiento_placebo"]), "Colapsa a cero"),
        ("Refutador: submuestra al 80 %", ms(rob["submuestra_80"]), "Estable"),
    ]
    return pd.DataFrame(filas, columns=["Estimación o prueba", "ATE", "Interpretación"])


def tabla6(entorno: dict) -> pd.DataFrame:
    libs = entorno["librerias"]
    filas = [
        ("Sistema operativo", entorno["so"]),
        ("Arquitectura / núcleos lógicos", f"{entorno['arquitectura']} / {entorno['nucleos_logicos']}"),
        ("Python", f"{entorno['python']} ({entorno['implementation']})"),
        *[(nombre, version) for nombre, version in libs.items()],
    ]
    return pd.DataFrame(filas, columns=["Elemento", "Valor"])


def tabla2(config: dict) -> pd.DataFrame:
    ga, dml, data, score = config["ga"], config["dml"], config["data"], config["score"]
    filas = [
        ("Datos", "Red / observaciones", f"{data['network']} / {data['n_samples']}"),
        ("Datos", "Tratamiento / resultado", f"{data['treatment']} / {data['outcome']}"),
        ("Score", "Penalización BIC / regularización", f"{score['penalty']} / {score['ridge']}"),
        ("AG", "Población / generaciones", f"{ga['population_size']} / {ga['n_generations']}"),
        ("AG", "Grado de entrada máximo", ga["max_indegree"]),
        ("AG", "p. cruce / p. mutación", f"{ga['p_crossover']} / {ga['p_mutation']}"),
        ("AG", "Torneo / elitismo", f"{ga['tournament_size']} / {ga['elitism']}"),
        ("Double ML", "Particiones / árboles / profundidad",
         f"{dml['n_splits']} / {dml['n_estimators']} / {dml['max_depth']}"),
        ("Diseño", "Réplicas / semilla maestra",
         f"{config['n_replications']} / {config['seed']}"),
    ]
    return pd.DataFrame(filas, columns=["Componente", "Parámetro", "Valor"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "multiseed.yaml"))
    parser.add_argument("--results", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    results_dir = ROOT / config.output_dir
    path = Path(args.results) if args.results else results_dir / f"multiseed_{config.n_replications}.json"
    if not path.exists():
        candidatos = sorted(results_dir.glob("multiseed_*.json"))
        if not candidatos:
            raise FileNotFoundError(
                f"No encontré resultados en {results_dir}. Ejecuta antes "
                "'python scripts/run_multiseed.py --reps 30'.")
        path = candidatos[-1]

    data = json.loads(path.read_text(encoding="utf-8"))
    resumen, entorno = data["resumen"], data["entorno"]

    tablas = {
        "tabla2_hiperparametros": tabla2(data["configuracion"]),
        "tabla3_recuperacion_estructural": tabla3(resumen),
        "tabla4_conjunto_ajuste": tabla4(resumen),
        "tabla5_efecto_y_robustez": tabla5(resumen),
        "tabla6_entorno": tabla6(entorno),
    }

    salida = results_dir / "tablas_documento"
    salida.mkdir(parents=True, exist_ok=True)
    titulos = {
        "tabla2_hiperparametros": "Tabla 2. Hiperparámetros del algoritmo genético y del estimador.",
        "tabla3_recuperacion_estructural": "Tabla 3. Recuperación estructural por método (30 réplicas).",
        "tabla4_conjunto_ajuste": "Tabla 4. Calidad del conjunto de ajuste y consecuencia estimacional.",
        "tabla5_efecto_y_robustez": "Tabla 5. Estimación del efecto causal y pruebas de robustez.",
        "tabla6_entorno": "Tabla 6. Entorno de ejecución y versiones de las librerías.",
    }
    for nombre, df in tablas.items():
        df.to_csv(salida / f"{nombre}.csv", index=False, encoding="utf-8")
        print("\n" + "=" * 78)
        print(titulos[nombre])
        print("=" * 78)
        print(df.to_string(index=False))

    print(f"\n[OK] {len(tablas)} tablas guardadas en: {salida}")
    print("Nota: la Tabla 1 (comparación conceptual de enfoques) es de elaboración "
          "propia y no se deriva de los datos.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
