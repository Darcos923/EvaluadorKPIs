import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from pprint import pprint

# ───────── helpers ───────────────────────────────────────────
BOOL = lambda x: str(x).strip().lower() == "true"
TXT = lambda node, default="": (
    (node.text or default).strip() if node is not None else default
)


def attr(node, key, default=None, cast=str):
    """Devuelve nodo.attrib[key] casteado, o default si falta nodo/atributo."""
    if node is None:
        return default
    val = node.attrib.get(key, default)
    return cast(val) if val is not None and cast is not str else val


# ───────── extractor ────────────────────────────────────────
def read_what_to_build(cfx_path: str | Path) -> dict:
    """Devuelve todos los parámetros de la pestaña 'What to build'."""
    with zipfile.ZipFile(Path(cfx_path)) as zf:
        xml_bytes = zf.read("config.xml")
    root = ET.fromstring(xml_bytes)

    wtb = root.find(".//WhatToBuild")
    if wtb is None:
        raise ValueError("No se encontró el bloque <WhatToBuild> dentro del .cfx")

    # 1) cabecera
    params = {
        "Strategy type": attr(
            wtb.find("StrategyType"), "type"
        ),  # simple | multiTF | template | improve
        "Trading direction": attr(
            wtb.find("MarketSides"), "type"
        ),  # long | short | both
        "Strategy Style": "SQX Signals",  # SQXSignals, PriceAction…
    }

    # 2) build-mode
    bmode = wtb.find("BuildMode")
    params["Build mode"] = {
        "Generation type": attr(
            bmode, "generationType"
        ),  # genetic-evolution, random-walk, …
        "Max generations": int(TXT(bmode.find("MaxGenerations"), 0)),
        "Population size": int(TXT(bmode.find("PopulationSize"), 0)),
        "Nº islands": int(TXT(bmode.find("Islands"), 0)),
        "Restart on finish": BOOL(TXT(bmode.find("RestartOnFinish"), "false")),
    }

    # 3) rules-complexity  (#condiciones, periodos, shifts…)
    chart = wtb.find("RulesComplexity/Chart")
    params["Rules of complexity"] = {
        "Min conditions": attr(chart, "minConditions", 0, int),
        "Max conditions": attr(chart, "maxConditions", 0, int),
        "Min exit conditions": attr(chart, "minExitConditions", 0, int),
        "Max exit conditions": attr(chart, "maxExitConditions", 0, int),
        "Min period": attr(chart, "minPeriod", 0, int),
        "Max period": attr(chart, "maxPeriod", 0, int),
        "Min shift": attr(chart, "minShift", 0, int),
        "Max shift": attr(chart, "maxShift", 0, int),
    }

    # 4) SL / PT  (ATR multiples, required flags…)
    slpt = wtb.find("SLPTOptions")
    slatr = BOOL(TXT(slpt.find("SLATR"), "false"))
    ptatr = BOOL(TXT(slpt.find("PTATR"), "false"))

    params["Stop loss"] = {
        "Required": BOOL(TXT(slpt.find("SLRequired"), "false")),
        "ATR based": slatr,
        "ATR multiple": (
            {
                "Min": float(TXT(slpt.find("MinSLATRMultiple"), 0)),
                "Max": float(TXT(slpt.find("MaxSLATRMultiple"), 0)),
            }
            if slatr
            else None
        ),
    }
    params["Profit target"] = {
        "Required": BOOL(TXT(slpt.find("PTRequired"), "false")),
        "ATR based": ptatr,
        "ATR bultiple": (
            {
                "Min": float(TXT(slpt.find("MinPTATRMultiple"), 0)),
                "Max": float(TXT(slpt.find("MaxPTATRMultiple"), 0)),
            }
            if ptatr
            else None
        ),
    }

    return params


# ───────── demo rápida ─────────
if __name__ == "__main__":
    cfg = read_what_to_build("../Build strategies.cfx")  # pon aquí tu .cfx
    pprint(cfg, sort_dicts=False)
