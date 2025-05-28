import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from pprint import pprint


TEST_PREC_MAP = {
    0: "Every tick (slower)",
    1: "Selected timeframe only (faster)",
    2: "Open prices only",
}


def read_data_settings(cfx_path: str | Path) -> dict:
    """Lee el bloque <Data> de un .cfx y devuelve todos los parámetros relevantes."""
    cfx_path = Path(cfx_path)

    # ── 1) descomprimir y leer config.xml ───────────────────────────────────
    with zipfile.ZipFile(cfx_path) as zf:
        xml_bytes = zf.read("config.xml")

    root = ET.fromstring(xml_bytes)
    data = root.find(".//Data")  # Settings/Data

    if data is None:
        raise ValueError("El .cfx no contiene bloque <Data>")

    # ── 2) BACKTEST SETUP (solo tomo el primero; añade un loop si tienes varios) ──
    setup = data.find("Setups/Setup")
    if setup is None:
        raise ValueError("No se encontró ningún <Setup> en el bloque <Data>")

    chart = setup.find("Chart")  # símbolo, TF, spread
    commissions_node = setup.find("Commissions/Method[@use='true']")
    swap_node = setup.find("Swap")

    params = {
        # --- Trading engine ---
        "Trading engine": setup.attrib.get("engine"),
        # --- Símbolo / Marco temporal / Fechas ---
        "Símbolo": chart.attrib.get("symbol"),
        "Temporalidad": chart.attrib.get("timeframe"),
        "Date from": setup.attrib.get("dateFrom"),
        "Date to": setup.attrib.get("dateTo"),
        # --- Parámetros de la prueba ---
        "Precisión de test": TEST_PREC_MAP.get(
            int(setup.attrib.get("testPrecision", 0))
        ),
        "Spread (pips)": int(chart.attrib.get("spread", 0)),
        "Slippage (pips)": int(setup.attrib.get("slippage", 0)),
        "Distancia mínima (pips)": int(setup.attrib.get("minDist", 0)),
        # --- Comisiones ---
        "Tipo de comisión": (
            commissions_node.attrib.get("type")
            if commissions_node is not None
            else None
        ),
        "Valor de comisión": (
            float(commissions_node.find("Params/Param").text)
            if commissions_node is not None
            else None
        ),
        # --- Swap ---
        "Swap habilitado": swap_node.attrib.get("use") == "true",
        "Tipo de swap": swap_node.attrib.get("type"),
        "Swap largo": float(swap_node.attrib.get("long", 0)),
        "Swap corto": float(swap_node.attrib.get("short", 0)),
        "Triple swap activo": swap_node.attrib.get("tripleSwapOn"),
    }

    # ── 3) PARTES OOS (Data range parts) ────────────────────────────────────
    oos_parts = []
    for rng in data.findall("OutOfSample/Range"):
        oos_parts.append(
            {"Date from": rng.attrib["dateFrom"], "Date to": rng.attrib["dateTo"]}
        )

    params["OOS Ranges"] = oos_parts

    return params


# ────────────── EJEMPLO DE EJECUCIÓN ──────────────
if __name__ == "__main__":
    datos = read_data_settings("../Build strategies.cfx")  # pon aquí tu ruta
    pprint(datos, sort_dicts=False)
