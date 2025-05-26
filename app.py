import json
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# ---------- KPI metadata -------------------------------------------------- #
KPI_COLS = {
    "Stability": ["Stability (OOS)", "Stability (IS)", "Stability"],
    "Ret/DD Ratio": ["Ret/DD Ratio (OOS)", "Ret/DD Ratio (IS)", "Ret/DD Ratio"],
    "Profit factor": ["Profit factor (OOS)", "Profit factor (IS)", "Profit factor"],
    "Win Rate": ["Win Rate (%)", "Winning Percent"],
    "CAGR/Max DD %": ["CAGR/Max DD % (OOS)", "CAGR/Max DD % (IS)", "CAGR/Max DD %"],
    "Win/Loss ratio": ["Win/Loss ratio (OOS)", "Win/Loss ratio (IS)", "Win/Loss ratio"],
    "Max DD %": ["Max DD % (OOS)", "Max DD % (IS)", "Max DD %"],
    "Sharpe Ratio": ["Sharpe Ratio (OOS)", "Sharpe Ratio (IS)", "Sharpe Ratio"],
}

KPI_INFO = {
    "Stability": ("Medida de suavidad de la curva de equity.", "Alto"),
    "Ret/DD Ratio": ("Retorno anualizado dividido por drawdown máximo.", "Alto"),
    "Profit factor": ("Ganancias totales / Pérdidas totales.", "Alto (>1)"),
    "Win Rate": ("% de operaciones ganadoras.", "Alto"),
    "CAGR/Max DD %": ("CAGR dividido por drawdown (%).", "Alto"),
    "Win/Loss ratio": ("Tamaño medio de ganancia / Tamaño medio de pérdida.", "Alto"),
    "Max DD %": ("Máximo drawdown relativo al capital inicial.", "Bajo"),
    "Sharpe Ratio": ("Exceso de retorno dividido por volatilidad (riesgo).", "Alto"),
}

DEFAULT_WEIGHTS = {
    "Stability": 0.20,
    "Ret/DD Ratio": 0.25,
    "Profit factor": 0.15,
    "Win Rate": 0.15,
    "CAGR/Max DD %": 0.10,
    "Win/Loss ratio": 0.05,
    "Max DD %": 0.05,
    "Sharpe Ratio": 0.05,
}

DEFAULT_THRESHOLDS = {
    "Stability": 0.6,
    "Ret/DD Ratio": 5,
    "Profit factor": 1,
    "Win Rate": 0.40,
    "CAGR/Max DD %": 1.3,
    "Win/Loss ratio": 0.70,
    "Sharpe Ratio": 1.0,
}


# ---------- helpers ------------------------------------------------------- #
def _detect_sep(p: Path) -> str:
    first = p.read_text(encoding="utf-8", errors="ignore").splitlines()[0]
    return ";" if ";" in first else ","


def _load(p: Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(p, sep=_detect_sep(p), low_memory=False)
    df["Comparativa"] = label
    return df


def _choose_cols(df: pd.DataFrame) -> dict:
    return {
        k: next((c for c in v if c in df.columns), None)
        for k, v in KPI_COLS.items()
        if next((c for c in v if c in df.columns), None)
    }


def _numeric(df: pd.DataFrame, map_: dict) -> pd.DataFrame:
    out = df[["Comparativa"]].copy()
    for k, col in map_.items():
        out[k] = pd.to_numeric(
            df[col].astype(str).str.replace(",", "."), errors="coerce"
        )
    if "Win Rate" not in out and "Win/Loss ratio" in out:
        r = out["Win/Loss ratio"]
        out["Win Rate"] = r / (1 + r)
    return out


def _composite(means: pd.DataFrame, weights: dict) -> pd.Series:
    m = means.copy()
    if "Max DD %" in m.columns:
        m["Max DD %"] = -m["Max DD %"]
    valid = [k for k in weights if k in m.columns]
    z = (m[valid] - m[valid].mean()) / m[valid].std()
    return (z * pd.Series({k: weights[k] for k in valid})).sum(axis=1)


def _detect_sep(p: Path) -> str:
    first = p.read_text(encoding="utf-8", errors="ignore").splitlines()[0]
    return ";" if ";" in first else ","


def _load_df(uploaded) -> pd.DataFrame:
    tmp = Path(f"temp_{uploaded.name}")
    tmp.write_bytes(uploaded.read())
    df = pd.read_csv(tmp, sep=_detect_sep(tmp), low_memory=False)
    return df


# ---------- Streamlit App ------------------------------------------------ #
st.title("Comparativa SQ | KPIs 📈")

# Pestañas: KPIs vs Robustez
tab1, tab2 = st.tabs(["Comparativa KPIs", "Pruebas de Robustez"])

# Sidebar común para KPI y Robustez
st.sidebar.header("Archivos KPI 🗂️")
kpi_files = st.sidebar.file_uploader(
    "Selecciona archivos CSV de KPI", type="csv", accept_multiple_files=True
)
st.sidebar.header("Archivos Robustez 🗂️")
robust_files = {
    "Builder": st.sidebar.file_uploader("CSV **Builder**", type="csv", key="rob_init"),
    "OOS": st.sidebar.file_uploader("CSV **OOS**", type="csv", key="rob_oos"),
    "HBP + MC Retest": st.sidebar.file_uploader(
        "CSV **HBP + MC Retest**", type="csv", key="rob_hbp"
    ),
    "Real Tick": st.sidebar.file_uploader(
        "CSV **Real Tick**", type="csv", key="rob_real"
    ),
    "SPP": st.sidebar.file_uploader("CSV **SPP**", type="csv", key="rob_spp"),
}
with tab1:
    # Sidebar: cargar archivos
    if kpi_files:
        labels = [Path(f.name).stem for f in kpi_files]

        # Sidebar: pesos y umbrales ajustables
        weights = {}
        st.sidebar.header("Ajustar pesos del Composite")
        for k, default in DEFAULT_WEIGHTS.items():
            weights[k] = st.sidebar.slider(
                k, min_value=0.0, max_value=1.0, value=float(default), step=0.01
            )

        thresholds = {}
        st.sidebar.header("Ajustar umbrales de filtros")
        for k, default in DEFAULT_THRESHOLDS.items():
            thresholds[k] = st.sidebar.number_input(f"Umbral {k}", value=float(default))

        # Procesamiento
        raw = []
        for f, lab in zip(kpi_files, labels):
            tmp = Path(f"temp_{lab}.csv")
            tmp.write_bytes(f.read())
            raw.append(_load(tmp, lab))
        colmap = _choose_cols(raw[0])
        df = pd.concat([_numeric(r, colmap) for r in raw], ignore_index=True)

        kpis = [k for k in KPI_COLS if k in df.columns]
        Comparativas = df.Comparativa.unique()

        # Estadísticas
        stats_df = (
            df.groupby("Comparativa")[kpis]
            .agg(["mean", "median", "std", "min", "max"])
            .round(3)
        )
        means_only = stats_df.xs("mean", level=1, axis=1)
        comp = _composite(means_only, weights).round(3)
        winner = comp.idxmax()

        # Mostrar KPI por KPI
        for kpi in kpis:
            st.header(kpi)
            desc, bias = KPI_INFO.get(kpi, ("", ""))
            st.markdown(f"> {desc} ➡️ **Mejor:** {bias}")

            # Tabla de estadísticas
            tbl = (
                stats_df[kpi]
                .reset_index()
                .rename(
                    columns={
                        "mean": "Media",
                        "median": "Mediana",
                        "std": "Desv.Std",
                        "min": "Mín",
                        "max": "Máx",
                    }
                )
            )
            st.table(tbl.set_index("Comparativa"))

            # Histograma
            fig, ax = plt.subplots(figsize=(5, 3))
            for b in Comparativas:
                ax.hist(
                    df[df.Comparativa == b][kpi].dropna(),
                    bins=30,
                    alpha=0.5,
                    label=f"B{b}",
                )
            ax.set_title(kpi)
            ax.set_xlabel("Rango de valores")
            ax.set_ylabel("Frecuencia")
            ax.legend(fontsize=6)
            st.pyplot(fig)

        # Composite
        st.header("Composite Score")
        st.markdown("> Z-score ponderado de todos los KPIs")
        st.subheader("Pesos usados en el Composite")
        st.write({k: f"{v:.2%}" for k, v in weights.items() if k in means_only.columns})

        comp_df = comp.reset_index()
        comp_df.columns = ["Comparativa", "Composite"]
        st.table(comp_df.set_index("Comparativa"))
        st.markdown(f"**Ganador comparativa global:** **🏆 {winner} 🏆**")

        # Filtros
        if thresholds:
            masks = [df[k] >= thr for k, thr in thresholds.items() if k in df.columns]
            if masks:
                df["passes"] = np.logical_and.reduce(masks)
                pct = (df.groupby("Comparativa")["passes"].mean() * 100).round(1)
                st.header("Estrategias que superan todos los filtros")
                st.markdown("**Filtros aplicados:**")
                for k, v in thresholds.items():
                    st.markdown(f"- {k} ≥ {v}")
                pct_df = pct.reset_index()
                pct_df.columns = ["Comparativa", "% de estrategias que pasan"]
                st.table(pct_df.set_index("Comparativa"))

# --- Tab 2: pruebas de robustez ---
with tab2:
    st.header("Pruebas de Robustez")
    # Usamos los archivos de sidebar
    init = robust_files["Builder"]
    if not init:
        st.info("Sube el Builder CSV para empezar la comparativa de robustez.")
    else:
        # Cálculo en flujo: Builder -> OOS -> HBP+MC -> Real Tick -> SPP
        st.markdown("> Builder ➡️ OOS ➡️ HBP+ MC Retest ➡️ Real Tick ➡️ SPP")
        steps = [
            ("Builder", init),
            ("OOS", robust_files.get("OOS")),
            ("HBP + MC Retest", robust_files.get("HBP + MC Retest")),
            ("Real Tick", robust_files.get("Real Tick")),
            ("SPP", robust_files.get("SPP")),
        ]
        flow = []
        prev_count = None
        for name, f in steps:
            if f:
                df = _load_df(f)
                cnt = len(df)
                if prev_count is None:
                    pct = "N/A"
                else:
                    pct = f"{cnt/prev_count*100:.1f}%" if prev_count > 0 else "N/A"
                flow.append({"Step": name, "Nº Estrategias": cnt, "% pasadas": pct})
                prev_count = cnt
            else:
                # saltar si no existe el archivo para ese step
                continue
        # Mostrar tabla de flujo
        flow_df = pd.DataFrame(flow).set_index("Step")
        st.table(flow_df)
