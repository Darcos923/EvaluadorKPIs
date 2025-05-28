import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from pprint import pprint


# ───────────────── helpers ────────────────────────────────────────────
def _bool(x: str | None) -> bool:
    return (x or "").strip().lower() == "true"


def _attr(node, name, default=None, cast=str):
    """attr helper – soporta nodos None y convierte al tipo deseado."""
    if node is None:
        return default
    val = node.attrib.get(name, default)
    return cast(val) if val is not None and cast is not str else val


# ───────────────── extractor ──────────────────────────────────────────
def read_ranking_settings(cfx_path: str | Path) -> dict:
    """Devuelve un dict con todos los parámetros de la pestaña Ranking."""
    # 1) leer config.xml
    with zipfile.ZipFile(Path(cfx_path)) as zf:
        xml = zf.read("config.xml")
    root = ET.fromstring(xml)

    # 2) localizar bloque:  <Rankings> (nuevo)  o  <Ranking> (viejo)
    block = root.find(".//Rankings") or root.find(".//Ranking")
    if block is None:
        raise ValueError("No se encontró ningún bloque <Rankings>/<Ranking> en el .cfx")

    results: dict = {}

    # • Máximo de estrategias
    results["Max strategies to store"] = int(block.findtext("MaxStrategies", 0))

    # • Stop generation
    stop = block.find("StopCondition")
    results["Stop generation"] = {
        "Type": _attr(stop, "type"),
        "Passed strategies": _attr(stop, "passedStrategies", 0, int),
        "Restart count": _attr(stop, "restartCount", 0, int),
        "Days": _attr(stop, "days", 0, int),
        "Hours": _attr(stop, "hours", 0, int),
        "Minutes": _attr(stop, "minutes", 0, int),
    }

    # • Fitness / ranking quality
    fit = block.find("FitnessCriteria/Settings/Ranking")
    results["Fitness type"] = {
        "Type": _attr(fit, "type"),  # Weighted, NetProfit, …
        # 'use' (Main data backtest / otros) se define arriba en <FitnessCriteria>
        "Method": _attr(block.find("FitnessCriteria"), "method"),
    }

    # • Criterios (solo los activos: use="true")
    crits = []
    for goal in fit.findall("Goal"):
        if not _bool(_attr(goal, "use", "false")):
            continue
        crits.append(
            {
                "Name": _attr(goal, "type"),
                "Type": "maximize" if goal.get("valueType") == "1" else "minimize",
                "Weight": _attr(goal, "weight", 0, int),
                "Target": _attr(goal, "target", 0, float),
            }
        )
    results["Ranking criteria"] = crits

    # • Filtros personalizados (panel derecho)
    custom_filters = []
    for idx, cond in enumerate(block.findall("Conditions/Condition[@use='true']")):
        left = cond.find(".//Column-Value").attrib.get("column")
        op = cond.find("Comparator").attrib.get("value")
        right = float(cond.find(".//Numeric-Value").attrib.get("value"))
        custom_filters.append({f"Condition {idx+1}": f"{left} {op} {right}"})
    results["Custom filters"] = custom_filters

    # • Automáticos y cross-check
    auto = block.find("AutomaticDismissal")
    results["Automatic filters ON"] = not _bool(auto.attrib.get("warnings", "false"))
    results["Cross check filters ON"] = _bool(
        _attr(block.find("ConditionsType"), None, "0")
    )  # 1 => CC activos

    # • Fit to portfolio
    port = block.find("FitPortfolio")
    results["Portfolio fit"] = {
        "Active": _attr(port, "active", "false") == "true",
        "Databank": _attr(port, "databank"),
        "Correlation threshold": _attr(port.find("Correlation"), "max", 0.0, float),
        "Timeframe": _attr(port.find("Correlation"), "period"),
    }

    return results


# ───────────── demo rápida ─────────────
if __name__ == "__main__":
    cfg = read_ranking_settings("../Build strategies.cfx")  # <- tu archivo
    pprint(cfg, sort_dicts=False)
