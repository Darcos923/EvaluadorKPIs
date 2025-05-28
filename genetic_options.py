import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from pprint import pprint


def _to_bool(txt: str | None) -> bool:
    return (txt or "").strip().lower() == "true"


def _safe_text(node, default=""):
    return node.text.strip() if node is not None and node.text else default


def read_genetic_settings(cfx_path: str | Path) -> dict:
    """Devuelve TODAS las opciones del bloque <BuildMode generationType="genetic-evolution">."""
    cfx_path = Path(cfx_path)

    # 1. Descomprimir y leer config.xml
    with zipfile.ZipFile(cfx_path) as zf:
        xml_bytes = zf.read("config.xml")

    # 2. Parsear XML y localizar el bloque <BuildMode generationType="genetic-evolution">
    root = ET.fromstring(xml_bytes)
    build_mode = root.find(
        ".//WhatToBuild/BuildMode[@generationType='genetic-evolution']"
    )

    if build_mode is None:
        raise ValueError("No se encontró el bloque genetic-evolution en el .cfx")

    # 3. Mapear directamente los nodos más simples --------------------------
    def _int(tag):
        return int(_safe_text(build_mode.find(tag), 0))

    def _flt(tag):
        return float(_safe_text(build_mode.find(tag), 0.0))

    def _pct(tag):
        return _int(tag)  # vienen ya como “80”  →  80 %

    params = {
        # Genetic options
        "Max generations": _int("MaxGenerations"),
        "Population size": _int("PopulationSize"),  # por isla
        "Crossover probability (%)": _pct("CrossoverProbability"),
        "Mutation probability (%)": _pct("MutationProbability"),
        # Islands options
        "Number of islands": _int("Islands"),
        "Migrate every X generations": _int("MigrationModulo"),
        "Migration rate (%)": _pct("MigrationRate"),
        # Initial population generation
        "Initial generation type": _int(
            "InitGenerationType"
        ),  # 0 = random, 1 = required size…
        "Decimation coefficient": _flt("DecimationCoef"),
        # Fresh blood
        "Replace similar fresh blood": _to_bool(
            _safe_text(build_mode.find("FreshBloodReplaceSimilar"))
        ),
        "Replace weakest fresh blood": _to_bool(
            _safe_text(build_mode.find("FreshBloodReplaceWeakest"))
        ),
        "Fresh blood weakest %": (
            _pct("FreshBloodWeakestPct")
            if build_mode.find("FreshBloodWeakestPct") is not None
            else None
        ),
        "Fresh blood weakest every N generations": (
            _int("FreshBloodWeakestGenerations")
            if build_mode.find("FreshBloodWeakestGenerations") is not None
            else None
        ),
        "Show last generation databank": _to_bool(
            _safe_text(build_mode.find("ShowLastGenerationDatabank"))
        ),
        # Evolution management
        "Restart on finish": _to_bool(
            build_mode.find("EvoRestartOnFinish").get("status")
        ),
        "Restart on stagnation": _to_bool(
            build_mode.find("EvoRestartOnStagnation").get("status")
        ),
        "Stagnation generations": int(
            build_mode.find("EvoRestartOnStagnation").get("generations")
        ),
        "Stagnation fitness type": int(
            build_mode.find("EvoRestartOnStagnation").get("fitnessType")
        ),
        # Sample split
        "In-sample ratio (%)": int(build_mode.find("EvoInSamplePeriod").get("ratio")),
    }

    # 4. Tabla “Filter generated initial population” ------------------------
    #    Cada fila aparece como <Condition use="true">…</Condition>
    filt = []
    for idx, cond in enumerate(build_mode.find("Conditions")):
        if cond.attrib.get("use") != "true":
            continue
        left = cond.find(".//Column-Value").attrib.get(
            "column"
        )  # ProfitFactor, NumberOfTrades…
        comp = cond.find(".//Comparator").attrib.get("value")  # >, <, >=, etc.
        right = cond.find(".//Numeric-Value").attrib.get("value")  # número del criterio
        filt.append({f"Condition {idx+1}": f"{left} {comp} {right}"})

    params["Initial population filters"] = filt

    return params


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Ejemplo de uso
    datos = read_genetic_settings("../Build strategies.cfx")
    pprint(datos, sort_dicts=False)
