#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bivariate (partial) correlation analysis between OFF-state kinematics
(FT & HM) and seiten-spezifischen Putamen-Z-Scores (DaT-SPECT) auf Basis
der hand-zentrierten Mastertabelle:
    Output/Data_Processed/final_merged_data.csv

Features:
- nutzt NUR final_merged_data.csv
- zwei Analyse-Modi:
    - "all": gesamte OFF-Kohorte
    - "pd_only": nur idiopathische PD (PD+ explizit ausgeschlossen)
- Age-Kontrolle über partielle Korrelation (Age aus Mastertabelle)
- FDR-Korrektur pro Side & Mode, aber getrennt pro Task (ft/hm)
- Multi-Panel Figure 1 pro Side & Mode:
    Panel A:
        - Contralateral + all: r_all vs r_pd_only Scatter
        - sonst: Heatmap der |r| signifikanter Kinematik-Features
    Panel B: FT vs HM r-Konzordanz
    Panel C: Top age-kontrollierte Scatterplots (Residuals)
"""

import os
import sys
import re
import unicodedata
import difflib

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import t as t_dist
from scipy.stats import pearsonr
from statsmodels.formula.api import ols


# -------------------------------------------------------------------------
# Projektstruktur & Dateien
# -------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))

INPUT_CSV = os.path.join(
    PROJECT_ROOT, "Output", "Data_Processed", "final_merged_data.csv"
)

DATA_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "Output", "Data")
PLOTS_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "Output", "Plots")
os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_OUTPUT_DIR, exist_ok=True)

OFF_LABEL = "off"

# -------------------------------------------------------------------------
# Analyse-Modi & Diagnose-Filter
# -------------------------------------------------------------------------

ANALYSIS_MODES = ["all", "pd_only"]
DIAGNOSE_COL = "Diagnose"

# idiopathische PD (case-insensitive)
DIAGNOSE_PD_PATTERNS = [
    "pd",               # exakter Label "PD"
    "parkinson",
    "morbus parkinson",
]

# explizite Ausschlüsse – z.B. PD+ (atypische Syndrome)
PD_EXCLUDE_PATTERNS = [
    "pd+",
]


def _normalize_diag_text(text: str) -> str:
    """Lowercase, strip, remove accents, keep only alnum/space."""
    text = str(text)
    if text is None:
        return ""
    text = text.strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if ch.isalnum() or ch.isspace())
    return text


def is_probable_pd_diagnosis(
    text: str,
    patterns=DIAGNOSE_PD_PATTERNS,
    exclude_patterns=PD_EXCLUDE_PATTERNS,
    min_similarity: float = 0.8,
) -> bool:
    """
    Heuristik, ob Diagnose idiopathische PD ist.

    - Erst harte Ausschlüsse (z.B. 'PD+')
    - Dann Substring- oder fuzzy-Matches auf PD-Patterns.
    """
    raw = str(text).strip().lower()
    if raw == "" or raw == "nan":
        return False

    # harte Ausschlüsse zuerst
    for neg in exclude_patterns:
        if neg in raw:
            return False

    norm_text = _normalize_diag_text(text)
    if norm_text == "":
        return False

    for pat in patterns:
        norm_pat = _normalize_diag_text(pat)

        # direkter Substring
        if norm_pat in norm_text:
            return True

        # fuzzy Match bei längeren Strings
        if len(norm_text) >= 4 and len(norm_pat) >= 4:
            ratio = difflib.SequenceMatcher(None, norm_text, norm_pat).ratio()
            if ratio >= min_similarity:
                return True

    return False


# -------------------------------------------------------------------------
# Analyse-Parameter
# -------------------------------------------------------------------------

CONTROL_FOR_AGE = True
AGE_COL = "Age"

SIDES_TO_ANALYZE = ["Contralateral", "Ipsilateral"]

BASE_KINEMATIC_COLS = [
    "meanamplitude",
    "stdamplitude",
    "meanspeed",
    "stdspeed",
    "meanrmsvelocity",
    "stdrmsvelocity",
    "meanopeningspeed",
    "stdopeningspeed",
    "meanclosingspeed",
    "stdclosingspeed",
    "meancycleduration",
    "stdcycleduration",
    "rangecycleduration",
    "rate",
    "amplitudedecay",
    "velocitydecay",
    "ratedecay",
    "cvamplitude",
    "cvcycleduration",
    "cvspeed",
    "cvrmsvelocity",
    "cvopeningspeed",
    "cvclosingspeed",
]

TASK_PREFIXES = ["ft", "hm"]

ALPHA_FDR = 0.05
TOP_N_FOR_SCATTER_PANEL = 3  # pro Zeile in Panel C (bis zu 6 Plots total)


# -------------------------------------------------------------------------
# Statistik-Helfer
# -------------------------------------------------------------------------

def fdr_bh(pvals: np.ndarray) -> np.ndarray:
    """
    Benjamini-Hochberg FDR-Korrektur.
    """
    pvals = np.asarray(pvals, dtype=float)
    n = pvals.size
    qvals = np.full(n, np.nan, dtype=float)
    if n == 0:
        return qvals

    order = np.argsort(pvals)
    ranked = pvals[order]

    prev_q = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1.0
        p = ranked[i]
        q = p * n / rank
        if q > prev_q:
            q = prev_q
        prev_q = q
        qvals[order[i]] = q

    return qvals


def pearson_or_partial(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    z_col: str | None = None,
    control_for_age: bool = True,
) -> tuple[float, float, int]:
    """
    Pearson- oder partielle Korrelation r_xy.z (wenn control_for_age=True und z_col gesetzt).
    """
    if control_for_age and z_col is not None:
        use_cols = [x_col, y_col, z_col]
    else:
        use_cols = [x_col, y_col]

    data = df[use_cols].dropna()
    n = data.shape[0]

    if n < (3 if (control_for_age and z_col is not None) else 2):
        return np.nan, np.nan, n

    if control_for_age and z_col is not None:
        r_xy = data[x_col].corr(data[y_col])
        r_xz = data[x_col].corr(data[z_col])
        r_yz = data[y_col].corr(data[z_col])

        denom = np.sqrt((1 - r_xz**2) * (1 - r_yz**2))
        if denom == 0 or np.isnan(denom):
            return np.nan, np.nan, n

        r = (r_xy - r_xz * r_yz) / denom

        dfree = n - 3
        if dfree <= 0 or abs(r) >= 1:
            p = np.nan
        else:
            t_val = r * np.sqrt(dfree / (1 - r**2))
            p = 2 * t_dist.sf(np.abs(t_val), dfree)
    else:
        r, p = pearsonr(data[x_col], data[y_col])

    return float(r), float(p), int(n)


# -------------------------------------------------------------------------
# Kinematik & Plot-Helfer
# -------------------------------------------------------------------------

def build_task_kinematic_cols(df: pd.DataFrame) -> dict:
    """
    Finde existierende Kinematikspalten pro Task (ft_*, hm_*).
    """
    task_cols = {}
    for task in TASK_PREFIXES:
        cols = [f"{task}_{base}" for base in BASE_KINEMATIC_COLS]
        cols_existing = [c for c in cols if c in df.columns]
        task_cols[task] = cols_existing
    return task_cols


def readable_kin_name(col_name: str) -> str:
    """
    'ft_meanamplitude' -> 'FT meanamplitude'
    """
    if "_" not in col_name:
        return col_name
    task, base = col_name.split("_", 1)
    return f"{task.upper()} {base}"


def base_from_kin(col_name: str) -> str:
    """
    Extrahiere Base-Kinematik ('ft_meanamplitude' -> 'meanamplitude').
    """
    return col_name.split("_", 1)[1] if "_" in col_name else col_name


def readable_side_name(side: str) -> str:
    return side.capitalize()


def _age_residuals(df: pd.DataFrame, x_col: str, age_col: str) -> pd.Series:
    """
    Residuen von x_col nach Regression auf Age (für Partialplot).
    """
    model = ols(f"{x_col} ~ {age_col}", data=df.dropna(subset=[x_col, age_col])).fit()
    return model.resid.reindex(df.index)


def compute_taskwise_correlations_for_subset(
    df_subset: pd.DataFrame,
    target_imaging_col: str,
    task_kin_cols: dict,
) -> pd.DataFrame:
    """
    Berechnet (partielle) Korrelationen zwischen allen Kinematik-Features und
    target_imaging_col für einen gegebenen Datensatz (z.B. PD-only).

    Gibt ein DataFrame mit Spalten:
        Task, Kinematic_Variable, Base, r, N
    zurück.
    """
    rows = []
    for task, cols in task_kin_cols.items():
        for kin_col in cols:
            if kin_col not in df_subset.columns:
                continue
            r, p, n = pearson_or_partial(
                df_subset,
                x_col=kin_col,
                y_col=target_imaging_col,
                z_col=AGE_COL if CONTROL_FOR_AGE else None,
                control_for_age=CONTROL_FOR_AGE,
            )
            rows.append(
                {
                    "Task": task,
                    "Kinematic_Variable": kin_col,
                    "Base": base_from_kin(kin_col),
                    "r": r,
                    "N": n,
                }
            )
    return pd.DataFrame(rows)


# -------------------------------------------------------------------------
# Figure 1: Multi-Panel
# -------------------------------------------------------------------------
def plot_figure1_multi_panel(
    sig_df: pd.DataFrame,
    res_df: pd.DataFrame,
    df_mode: pd.DataFrame,
    side: str,
    mode: str,
    target_imaging_col: str,
    out_dir: str,
    task_kin_cols: dict,
):
    """
    Multi-Panel Figure 1.

    Panel A:
        - Contralateral + all: r_all vs r_pd_only Scatterplot
        - sonst: Heatmap der |r| (nur signifikante Features)
    Panel B: FT vs HM Korrelation (alle Features, Signifikanzkodierung)
             + Regressionslinie
    Panel C: Top-|r|-Scatterplots (Age-residualized) + Regressionslinien
    """
    # Für Heatmap/Top-Plots nur signifikante Daten
    sig_df = sig_df.copy()
    sig_df["Base"] = sig_df["Kinematic_Variable"].apply(base_from_kin)
    sig_df["abs_r"] = sig_df["r"].abs()

    # Heatmap-Daten (Standardfall)
    panelA = sig_df.copy()
    panelA["Task"] = panelA["Task"].str.upper()
    heat = (
        panelA.pivot_table(
            index="Base",
            columns="Task",
            values="abs_r",
            aggfunc="max",
        )
        .fillna(0.0)
        .sort_index()
    )

    # Panel B: FT vs HM
    res_df = res_df.copy()
    res_df["Base"] = res_df["Kinematic_Variable"].apply(base_from_kin)
    panelB = (
        res_df.assign(Task=lambda d: d["Task"].str.lower())
        .pivot_table(
            index="Base",
            columns="Task",
            values="r",
            aggfunc=lambda x: x.loc[x.abs().idxmax()] if len(x.dropna()) > 0 else np.nan,
        )
    )

    # Signifikanz-Status auf Basis von sig_df
    sig_ft = sig_df.loc[sig_df["Task"] == "ft", "Base"].unique()
    sig_hm = sig_df.loc[sig_df["Task"] == "hm", "Base"].unique()

    def sig_status(base: str) -> str:
        in_ft = base in sig_ft
        in_hm = base in sig_hm
        if in_ft and in_hm:
            return "Significant: Both"
        if in_ft:
            return "Significant: FT"
        if in_hm:
            return "Significant: HM"
        return "Not significant"

    panelB = panelB.dropna(how="all")
    panelB["Significance"] = [sig_status(b) for b in panelB.index]

    palette = {
        "Significant: Both": "black",
        "Significant: FT": "#4c72b0",
        "Significant: HM": "#dd8452",
        "Not significant": "grey",
    }
    sizes = {
        "Significant: Both": 80,
        "Significant: FT": 60,
        "Significant: HM": 60,
        "Not significant": 30,
    }

    # Panel C: Top-|r|-Scatterplots
    sig_sorted = sig_df.sort_values("abs_r", ascending=False)
    max_plots = 2 * TOP_N_FOR_SCATTER_PANEL
    top_examples = sig_sorted.head(max_plots)

    # Figure-Layout
    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.4], hspace=0.35, wspace=0.3)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])

    gsC = gs[1, :].subgridspec(2, TOP_N_FOR_SCATTER_PANEL, hspace=0.5, wspace=0.4)
    axesC = [
        [fig.add_subplot(gsC[i, j]) for j in range(TOP_N_FOR_SCATTER_PANEL)]
        for i in range(2)
    ]

    # ------------------------------------------------------------------
    # Panel A – Spezialfall Contralateral + all: r_all vs r_pd_only
    # ------------------------------------------------------------------
    compare_pd = (side == "Contralateral") and (mode == "all")

    if compare_pd:
        if DIAGNOSE_COL in df_mode.columns:
            df_pd = df_mode[df_mode[DIAGNOSE_COL].apply(is_probable_pd_diagnosis)].copy()
        else:
            df_pd = pd.DataFrame()

        if df_pd.empty:
            # Fallback: normale Heatmap
            if not heat.empty:
                sns.heatmap(
                    heat,
                    ax=axA,
                    cmap="vlag",
                    center=0,
                    annot=True,
                    fmt=".2f",
                    cbar_kws={"label": "|r|"},
                )
                axA.set_title("A) |r| for significant kinematic features",
                              loc="left", fontsize=13, fontweight="bold")
                axA.set_xlabel("Task")
                axA.set_ylabel("Base kinematic feature")
            else:
                axA.text(
                    0.5,
                    0.5,
                    "No significant features for heatmap",
                    ha="center",
                    va="center",
                    transform=axA.transAxes,
                )
                axA.set_axis_off()
        else:
            # r_all aus res_df, r_pd_only neu berechnen
            df_all_r = (
                res_df[["Task", "Kinematic_Variable", "r", "q"]]
                .copy()
                .rename(columns={"r": "r_all"})
            )
            df_all_r["Base"] = df_all_r["Kinematic_Variable"].apply(base_from_kin)

            df_pd_r = compute_taskwise_correlations_for_subset(
                df_subset=df_pd,
                target_imaging_col=target_imaging_col,
                task_kin_cols=task_kin_cols,
            ).rename(columns={"r": "r_pd"})

            merged = pd.merge(
                df_all_r,
                df_pd_r[["Task", "Kinematic_Variable", "r_pd"]],
                on=["Task", "Kinematic_Variable"],
                how="inner",
            )
            merged = merged.dropna(subset=["r_all", "r_pd"])

            if merged.empty:
                axA.text(
                    0.5,
                    0.5,
                    "No overlap between all and PD-only\nfor correlation estimates.",
                    ha="center",
                    va="center",
                    transform=axA.transAxes,
                )
                axA.set_axis_off()
            else:
                merged["Significant_all"] = merged["q"].notna() & (merged["q"] < ALPHA_FDR)

                palette_task = {"ft": "#4c72b0", "hm": "#dd8452"}
                marker_sig = {True: "o", False: "x"}
                size_sig = {True: 70, False: 40}

                for (task_name, sig_flag), grp in merged.groupby(["Task", "Significant_all"]):
                    axA.scatter(
                        grp["r_all"],
                        grp["r_pd"],
                        label=f"{task_name.upper()}, "
                              f"{'sig. in all' if sig_flag else 'n.s. in all'}",
                        s=size_sig[sig_flag],
                        marker=marker_sig[sig_flag],
                        alpha=0.8,
                        edgecolor="k",
                        linewidth=0.5,
                        c=palette_task.get(task_name, "grey"),
                    )

                lim_min = min(merged["r_all"].min(), merged["r_pd"].min()) - 0.05
                lim_max = max(merged["r_all"].max(), merged["r_pd"].max()) + 0.05

                # Diagonale
                axA.plot([lim_min, lim_max], [lim_min, lim_max],
                         linestyle="--", linewidth=1, color="black", label="Identity")

                # Regressionslinie r_pd ~ r_all
                # x = merged["r_all"].values
                # y = merged["r_pd"].values
                # if len(x) >= 3:
                #     slope, intercept = np.polyfit(x, y, 1)
                #     x_line = np.linspace(lim_min, lim_max, 100)
                #     y_line = slope * x_line + intercept
                #     axA.plot(x_line, y_line, linewidth=1.5, color="grey",
                #              label="Regression")

                axA.set_xlim(lim_min, lim_max)
                axA.set_ylim(lim_min, lim_max)

                axA.set_xlabel("Correlation r (all patients)")
                axA.set_ylabel("Correlation r (PD-only)")
                axA.set_title(
                    "A) Consistency of effect sizes\n"
                    "(all vs PD-only, age-controlled)",
                    loc="left", fontsize=13, fontweight="bold",
                )
                axA.legend(fontsize=8, loc="lower right")
    else:
        # Standard-Heatmap
        if not heat.empty:
            sns.heatmap(
                heat,
                ax=axA,
                cmap="vlag",
                center=0,
                annot=True,
                fmt=".2f",
                cbar_kws={"label": "|r|"},
            )
            axA.set_title("A) |r| for significant kinematic features",
                          loc="left", fontsize=13, fontweight="bold")
            axA.set_xlabel("Task")
            axA.set_ylabel("Base kinematic feature")
        else:
            axA.text(
                0.5,
                0.5,
                "No significant features for heatmap",
                ha="center",
                va="center",
                transform=axA.transAxes,
            )
            axA.set_axis_off()

    # ---- Panel B: FT vs HM Scatter + Regressionslinie ----
    if {"ft", "hm"}.issubset(panelB.columns):
        pb = panelB.dropna(subset=["ft", "hm"])
        if not pb.empty:
            for status, grp in pb.groupby("Significance"):
                axB.scatter(
                    grp["ft"],
                    grp["hm"],
                    label=status,
                    s=sizes.get(status, 40),
                    alpha=0.8,
                    edgecolor="k",
                    linewidth=0.5,
                    c=palette.get(status, "grey"),
                )

            # Achsenlimits
            lims = [
                min(pb["ft"].min(), pb["hm"].min()) - 0.05,
                max(pb["ft"].max(), pb["hm"].max()) + 0.05,
            ]

            # Diagonale
            axB.plot(lims, lims, linestyle="--", linewidth=1, color="black",
                     label="Identity")

            # Regressionslinie hm ~ ft
            x = pb["ft"].values
            y = pb["hm"].values
            if len(x) >= 3:
                slope, intercept = np.polyfit(x, y, 1)
                x_line = np.linspace(lims[0], lims[1], 100)
                y_line = slope * x_line + intercept
                axB.plot(x_line, y_line, linewidth=1.5, color="grey",
                         label="Regression")

            axB.set_xlim(lims)
            axB.set_ylim(lims)

            axB.set_xlabel("FT correlation (r)")
            axB.set_ylabel("HM correlation (r)")
            axB.set_title("B) FT vs HM correlation across base features",
                          loc="left", fontsize=13, fontweight="bold")
            axB.legend(fontsize=8, loc="lower right")
        else:
            axB.text(
                0.5,
                0.5,
                "Not enough data for FT/HM comparison",
                ha="center",
                va="center",
                transform=axB.transAxes,
            )
            axB.set_axis_off()
    else:
        axB.text(
            0.5,
            0.5,
            "FT/HM columns missing for panel B",
            ha="center",
            va="center",
            transform=axB.transAxes,
        )
        axB.set_axis_off()

    # ---- Panel C: Top-Scatterplots + Regressionslinien ----
    # globales "C)"-Label im Figure-Rand
    fig.text(0.02, 0.47, "C)",
             fontsize=14, fontweight="bold",
             va="center", ha="left", transform=fig.transFigure)

    if top_examples.empty:
        for row_axes in axesC:
            for ax in row_axes:
                ax.set_axis_off()
    else:
        for idx, (_, row) in enumerate(top_examples.iterrows()):
            row_idx = idx // TOP_N_FOR_SCATTER_PANEL
            col_idx = idx % TOP_N_FOR_SCATTER_PANEL
            if row_idx >= 2:
                break
            ax = axesC[row_idx][col_idx]

            kin_col = row["Kinematic_Variable"]
            r_val = row["r"]
            q_val = row["q"]
            n_val = row["N"]

            cols = [kin_col, target_imaging_col]
            if CONTROL_FOR_AGE:
                cols.append(AGE_COL)
            plot_data = df_mode[cols].dropna()

            if plot_data.shape[0] < 5:
                ax.set_axis_off()
                continue

            if CONTROL_FOR_AGE and AGE_COL in plot_data.columns:
                x_vals = _age_residuals(plot_data, kin_col, AGE_COL)
                y_vals = _age_residuals(plot_data, target_imaging_col, AGE_COL)
                ax.scatter(x_vals, y_vals, alpha=0.7, s=20, edgecolor="k", linewidth=0.5)

                # Regressionslinie img_resid ~ kin_resid
                x = x_vals.values
                y = y_vals.values
                if np.isfinite(x).sum() >= 3 and np.isfinite(y).sum() >= 3:
                    slope, intercept = np.polyfit(x, y, 1)
                    x_line = np.linspace(x.min(), x.max(), 100)
                    y_line = slope * x_line + intercept
                    ax.plot(x_line, y_line, linestyle="--", linewidth=1.5, color="grey")
                ax.set_xlabel(f"{readable_kin_name(kin_col)} (age-residualized)")
                ax.set_ylabel(f"{target_imaging_col} (age-residualized)")
            else:
                x = plot_data[kin_col].values
                y = plot_data[target_imaging_col].values
                ax.scatter(x, y, alpha=0.7, s=20, edgecolor="k", linewidth=0.5)
                if len(x) >= 3:
                    slope, intercept = np.polyfit(x, y, 1)
                    x_line = np.linspace(x.min(), x.max(), 100)
                    y_line = slope * x_line + intercept
                    ax.plot(x_line, y_line, linestyle="--", linewidth=1.5, color="grey")
                ax.set_xlabel(readable_kin_name(kin_col))
                ax.set_ylabel(target_imaging_col)

            title = f"Top {idx+1}: r={r_val:.2f}, q={q_val:.3f}, N={int(n_val)}"
            ax.set_title(title, fontsize=9)

        # restliche Achsen ausblenden
        for idx in range(len(top_examples), 2 * TOP_N_FOR_SCATTER_PANEL):
            row_idx = idx // TOP_N_FOR_SCATTER_PANEL
            col_idx = idx % TOP_N_FOR_SCATTER_PANEL
            if row_idx < 2:
                axesC[row_idx][col_idx].set_axis_off()

    fig.suptitle(
        f"Figure 1: Age-controlled bivariate findings for {readable_side_name(side)} Putamen "
        f"({mode}, OFF)",
        fontsize=16,
        fontweight="bold",
    )
    plt.subplots_adjust(top=0.93)

    filename = f"Figure1_Bivariate_Findings_{side}_Summary_AgeControlled_{mode}.png"
    out_path = os.path.join(out_dir, filename)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[SUCCESS] Multi-panel Figure 1 gespeichert: {os.path.abspath(out_path)}")


# -------------------------------------------------------------------------
# Hauptfunktion
# -------------------------------------------------------------------------

def main():
    print("\n" + "=" * 80)
    print("Starting bivariate analysis using master table 'final_merged_data.csv'")
    print("=" * 80 + "\n")

    # ---- Daten laden ----
    if not os.path.exists(INPUT_CSV):
        print(
            f"[FATAL ERROR] Input file not found at '{INPUT_CSV}'. "
            "Make sure Script I has been run successfully."
        )
        sys.exit(1)

    try:
        try:
            df_full = pd.read_csv(INPUT_CSV, sep=";", decimal=".")
        except Exception:
            df_full = pd.read_csv(INPUT_CSV, sep=",", decimal=".")
        print(f"[INFO] Original data loaded successfully. Shape: {df_full.shape}")
    except Exception as e:
        print(f"[FATAL ERROR] Error while loading data: {e}")
        sys.exit(1)

    required_cols = ["Medication Condition", "Hand_Performed"]
    missing_req = [c for c in required_cols if c not in df_full.columns]
    if missing_req:
        print(
            "[CRITICAL ERROR] Essential columns missing from the master table: "
            + ", ".join(missing_req)
        )
        sys.exit(1)

    # OFF-State filtern
    df_full["Medication Condition"] = (
        df_full["Medication Condition"].astype(str).str.strip().str.lower()
    )
    df = df_full[df_full["Medication Condition"] == OFF_LABEL].copy()
    if df.empty:
        print("[FATAL ERROR] No OFF-state data found. Exiting.")
        sys.exit(1)
    print(f"[INFO] Data restricted to OFF-state. Shape: {df.shape}")

    # Age-Control
    global CONTROL_FOR_AGE
    if CONTROL_FOR_AGE and AGE_COL not in df.columns:
        print("\n" + "#" * 60)
        print(f"### WARNING: Age control requested, but '{AGE_COL}' column not found! ###")
        print("### Reverting to standard Pearson correlation. ###")
        print("#" * 60 + "\n")
        CONTROL_FOR_AGE = False
    elif CONTROL_FOR_AGE:
        print(f"\n[INFO] Age control is ENABLED. Using '{AGE_COL}' as covariate.")
        before = df.shape[0]
        df = df.dropna(subset=[AGE_COL]).copy()
        after = df.shape[0]
        print(
            f"[INFO] Dropped {before - after} rows due to missing Age. Remaining: {after}."
        )
    else:
        print("\n[INFO] Age control is DISABLED. Using standard Pearson correlations.")

    # Kinematik-Spalten pro Task
    global task_kin_cols
    task_kin_cols = build_task_kinematic_cols(df)
    for task, cols in task_kin_cols.items():
        print(f"[INFO] Found {len(cols)} kinematic columns for task '{task}'.")
        if not cols:
            print(f"[WARNING] No kinematic columns found for task '{task}' in the master table.")

    # ------------------------------------------------------------------
    # Analysen: pro Mode & Side
    # ------------------------------------------------------------------
    for mode in ANALYSIS_MODES:
        print("\n" + "=" * 80)
        print(f"=== STARTING ANALYSIS MODE: {mode} ===")
        print("=" * 80 + "\n")

        # Mode-spezifischer Datensatz
        if mode == "all":
            df_mode = df.copy()

        elif mode == "pd_only":
            if DIAGNOSE_COL not in df.columns:
                print(
                    f"[WARNING] Column '{DIAGNOSE_COL}' not found. "
                    "Cannot run PD-only mode. Skipping."
                )
                continue

            pd_mask = df[DIAGNOSE_COL].apply(is_probable_pd_diagnosis)
            n_total = df.shape[0]
            n_pd = int(pd_mask.sum())
            n_missing_diag = int(df[DIAGNOSE_COL].isna().sum())

            print(
                f"[INFO] PD-only filter: {n_pd} / {n_total} rows flagged as probable idiopathic PD "
                f"({n_missing_diag} rows with missing '{DIAGNOSE_COL}')."
            )

            if n_pd == 0:
                print("[WARNING] No rows identified as PD for PD-only mode. Skipping.")
                continue

            df_mode = df[pd_mask].copy()

        else:
            print(f"[ERROR] Unknown analysis mode '{mode}'. Skipping.")
            continue

        if df_mode.empty:
            print(f"[WARNING] Mode '{mode}' resulted in an empty dataframe. Skipping.")
            continue

        # ---- Pro Seite ----
        for side in SIDES_TO_ANALYZE:
            print("\n" + "#" * 80)
            print(f"###   STARTING ANALYSIS FOR: {side.upper()} SIDE (mode={mode})   ###")
            print("#" * 80 + "\n")

            target_imaging_col = f"{side}_Putamen_Z"
            if target_imaging_col not in df_mode.columns:
                print(
                    f"[FATAL ERROR] Imaging column '{target_imaging_col}' not found "
                    f"in dataframe for mode={mode}. Skipping this side."
                )
                continue

            # ---------------- Cohort Summary ----------------
            try:
                summary_df = df_mode.copy()
                sex_col = "Sex"

                num_hand_obs = summary_df.shape[0]
                num_unique_patients = (
                    summary_df["Patient ID"].nunique()
                    if "Patient ID" in summary_df.columns
                    else np.nan
                )
                mean_age = (
                    summary_df[AGE_COL].mean() if AGE_COL in summary_df.columns else np.nan
                )
                std_age = (
                    summary_df[AGE_COL].std() if AGE_COL in summary_df.columns else np.nan
                )

                percent_male = np.nan
                if sex_col in summary_df.columns:
                    male_mask = (
                        summary_df[sex_col]
                        .astype(str)
                        .str.strip()
                        .str.lower()
                        .isin(["m", "male", "1"])
                    )
                    total_with_sex = summary_df[sex_col].notna().sum()
                    if total_with_sex > 0:
                        percent_male = float(male_mask.sum()) / float(total_with_sex) * 100.0
                else:
                    print(
                        f"[INFO] Sex column '{sex_col}' not found. "
                        "Sex-related summary statistics set to NaN."
                    )

                imaging_data = summary_df[target_imaging_col].dropna()
                mean_z = imaging_data.mean() if not imaging_data.empty else np.nan
                std_z = imaging_data.std() if not imaging_data.empty else np.nan
                min_z = imaging_data.min() if not imaging_data.empty else np.nan
                max_z = imaging_data.max() if not imaging_data.empty else np.nan

                summary_dict = {
                    "Metric": [
                        "Number of unique patients",
                        "Number of hand observations (N)",
                        "Mean Age (years)",
                        "Std Dev Age (years)",
                        "Percentage Male (%)",
                        f"Mean {side} Putamen Z-score",
                        f"Std Dev {side} Putamen Z-score",
                        f"Min {side} Putamen Z-score",
                        f"Max {side} Putamen Z-score",
                    ],
                    "Value": [
                        num_unique_patients,
                        num_hand_obs,
                        mean_age,
                        std_age,
                        percent_male,
                        mean_z,
                        std_z,
                        min_z,
                        max_z,
                    ],
                }

                summary_out = pd.DataFrame(summary_dict)
                summary_out["Value"] = summary_out["Value"].round(2)

                summary_path = os.path.join(
                    DATA_OUTPUT_DIR,
                    f"cohort_summary_statistics_{side.lower()}_{mode}.csv",
                )
                summary_out.to_csv(summary_path, index=False, sep=";", decimal=".")
                print(
                    f"[SUCCESS] Cohort summary (side={side}, mode={mode}) "
                    f"saved to: {os.path.abspath(summary_path)}"
                )
            except Exception as e:
                print(
                    f"[ERROR] Failed to generate cohort summary for side={side}, mode={mode}. "
                    f"Reason: {e}"
                )

            # ---------------- Bivariate Korrelationen ----------------
            corr_results = []

            for task in TASK_PREFIXES:
                kin_cols = task_kin_cols.get(task, [])
                if not kin_cols:
                    continue

                for kin_col in kin_cols:
                    if kin_col not in df_mode.columns:
                        continue

                    r, p, n = pearson_or_partial(
                        df_mode,
                        x_col=kin_col,
                        y_col=target_imaging_col,
                        z_col=AGE_COL if CONTROL_FOR_AGE else None,
                        control_for_age=CONTROL_FOR_AGE,
                    )

                    corr_results.append(
                        {
                            "Side": side,
                            "Mode": mode,
                            "Task": task,
                            "Kinematic_Variable": kin_col,
                            "Target_Variable": target_imaging_col,
                            "r": r,
                            "p": p,
                            "N": n,
                            "Controlled_for_Age": CONTROL_FOR_AGE,
                        }
                    )

            if not corr_results:
                print(
                    f"[WARNING] No correlation results computed for side={side}, mode={mode}."
                )
                continue

            res_df = pd.DataFrame(corr_results)

            # FDR-Korrektur pro Side & Mode, aber getrennt nach Task
            qvals = pd.Series(np.nan, index=res_df.index, dtype=float)
            for task_name, sub in res_df.groupby("Task"):
                mask_valid_p = sub["p"].notna()
                if not mask_valid_p.any():
                    continue
                pvals_task = sub.loc[mask_valid_p, "p"].values
                qvals_task = fdr_bh(pvals_task)
                qvals.loc[sub.index[mask_valid_p]] = qvals_task
            res_df["q"] = qvals.values

            # alle Resultate speichern
            raw_path = os.path.join(
                DATA_OUTPUT_DIR,
                f"all_raw_bivariate_results_{side.lower()}_{mode}.csv",
            )
            res_df.to_csv(raw_path, index=False, sep=";", decimal=".")
            print(
                f"[SUCCESS] Raw bivariate results (side={side}, mode={mode}) "
                f"saved to: {os.path.abspath(raw_path)}"
            )

            # signifikante Resultate (q < ALPHA_FDR)
            sig_df = res_df[(res_df["q"].notna()) & (res_df["q"] < ALPHA_FDR)].copy()
            sig_path = os.path.join(
                DATA_OUTPUT_DIR,
                f"all_significant_bivariate_results_{side.lower()}_{mode}.csv",
            )
            sig_df.to_csv(sig_path, index=False, sep=";", decimal=".")
            print(
                f"[SUCCESS] Significant results (q < {ALPHA_FDR}) "
                f"(side={side}, mode={mode}) saved to: {os.path.abspath(sig_path)}"
            )

            # Multi-Panel Figure 1:
            # - wenn keine signifikanten Befunde: für pd_only überspringen,
            #   für all trotzdem zeichnen (Trend-Figure)
            if sig_df.empty and mode == "pd_only":
                print(
                    f"[INFO] Keine signifikanten Ergebnisse für side={side}, mode={mode}; "
                    f"Figure 1 wird übersprungen."
                )
            else:
                try:
                    plot_figure1_multi_panel(
                        sig_df=sig_df if not sig_df.empty else res_df.copy(),
                        res_df=res_df,
                        df_mode=df_mode,
                        side=side,
                        mode=mode,
                        target_imaging_col=target_imaging_col,
                        out_dir=PLOTS_OUTPUT_DIR,
                        task_kin_cols=task_kin_cols,
                    )
                except Exception as e:
                    print(
                        f"[ERROR] Could not generate Figure 1 for side={side}, mode={mode}. "
                        f"Reason: {e}"
                    )

    print("\n--- Bivariate analysis script execution finished ---\n")


if __name__ == "__main__":
    main()
