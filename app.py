# -*- coding: utf-8 -*-
"""
Solvation NASA Polynomials — Streamlit Application
PR EoS + COSMO-RS

@author: Francisco Carlos Paes, 2026
Equipe Thermodynamique et Energie (ThermE)
Laboratoire Réactions et Génie des Procédés (LRGP)
UMR 7274 CNRS - Université de Lorraine
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tempfile
import os
import io
import time
import gc
import warnings

warnings.filterwarnings("ignore")

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

from sub_model import interaction_mtx
from sub_load_molecules import loadmolecules
from sub_properties import fit_nasa


# ═══════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════
#  Introduction
# ═══════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════

st.image('B2.png')


# ──────────────────────────────────────────────────────────────
# Page configuration
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Solvation NASA Polynomials",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────
# Cached resource loader (runs once per server session)
# ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Chargement de la base de données…")
def load_resources():
    psigmaGC = pd.read_csv("database/gcm/psigmaGC.csv", delimiter=";").to_numpy()
    ceosGC   = pd.read_csv("database/gcm/ceosGC.csv",   delimiter=";").to_numpy()

    sig_lim    = 0.03
    delta_sig  = 0.001
    sigma      = np.arange(-sig_lim, sig_lim + delta_sig, delta_sig)

    params = {
        "RSI":         8.31446261815324 / 1000.,   # kJ/(mol·K)
        "aeff":        4.73,                        # Å²
        "alpha":       10473.,                      # kJ/mol
        "Chb":         48321.,                      # kJ/mol
        "ChbT1":       20.,
        "ChbT2":       200.,
        "sigma_hb":    0.0109,                      # e/Å²
        "sigma_max":   0.03,                        # e/Å²
        "delta_sigma": 0.001,                       # e/Å²
        "q1":         -0.6232252401402305,
        "s":           1.5,
        "psigmaGC":    psigmaGC,
        "ceosGC":      ceosGC,
        "sigma":       sigma,
    }
    params["Em298"], params["Ehb298"] = interaction_mtx(params)

    BDD = joblib.load("database/eos/BDD.joblib")
    return params, BDD


# ──────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────
st.title("⚗️ Solvation NASA Polynomials")
st.markdown(
    "Calcul des polynômes NASA de solvatation avec le modèle **PR EoS + COSMO-RS**  \n"
    "*F.C. Paes — LRGP UMR 7274 CNRS, Université de Lorraine, 2026*  \n"
    "Molécules supportées : **C / H / O** (molécules stables et radicaux libres)"
)
st.divider()

# ──────────────────────────────────────────────────────────────
# Sidebar — system conditions (defaults from system.txt)
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Conditions du système")
    st.caption("Valeurs par défaut issues de `system.txt`")

    P_bar = st.number_input(
        "Pression (bar)", value=1.01325, min_value=0.001, format="%.5f"
    )
    T_min = st.number_input(
        "Température minimale (K)", value=273.15, min_value=50.0, max_value=3000.0, step=5.0
    )
    T_max = st.number_input(
        "Température maximale (K)", value=473.15, min_value=50.0, max_value=3000.0, step=5.0
    )
    n_T = st.number_input(
        "Database", value=21, min_value=5, max_value=200, step=1
    )

    st.divider()
    st.markdown("**Format des fichiers :**")
    st.markdown(
        "**Solvants** — première ligne : `smiles zi`  \n"
        "Lignes suivantes : `SMILES  fraction_molaire`\n\n"
        "**Solutés** — première ligne : `smiles`  \n"
        "Lignes suivantes : un SMILES par ligne"
    )
    with st.expander("Exemple — solvants"):
        st.code("smiles zi\nCCCCO 1.0", language=None)
    with st.expander("Exemple — solutés"):
        st.code("smiles\nC\nCO\nC=O\nC(=O)=O", language=None)

# ──────────────────────────────────────────────────────────────
# File upload area
# ──────────────────────────────────────────────────────────────
col_solv, col_solu = st.columns(2)

with col_solv:
    st.subheader("🧪 Fichier solvants")
    solvent_file = st.file_uploader(
        "Charger le fichier solvants (.txt)", type=["txt"], key="solvents"
    )
    if solvent_file is None:
        st.info(
            "Aucun fichier chargé.  \n"
            "**Solvant par défaut** : 1-butanol (`CCCCO`) pur."
        )
    else:
        content = solvent_file.read().decode("utf-8")
        solvent_file.seek(0)
        lines = [l for l in content.splitlines() if l.strip()]
        st.caption(f"{len(lines) - 1} solvant(s) détecté(s)")
        st.code(content[:400] + ("…" if len(content) > 400 else ""), language=None)

with col_solu:
    st.subheader("🔬 Fichier solutés")
    solute_file = st.file_uploader(
        "Charger le fichier solutés (.txt)", type=["txt"], key="solutes"
    )
    if solute_file is None:
        st.info(
            "Aucun fichier chargé.  \n"
            "**Solutés par défaut** : liste exemple (129 molécules C/H/O)."
        )
    else:
        content = solute_file.read().decode("utf-8")
        solute_file.seek(0)
        lines = [l for l in content.splitlines() if l.strip()]
        st.caption(f"{len(lines) - 1} soluté(s) détecté(s)")
        preview_lines = lines[:12]
        st.code(
            "\n".join(preview_lines) + ("\n…" if len(lines) > 12 else ""),
            language=None,
        )

st.divider()

# ──────────────────────────────────────────────────────────────
# Run button
# ──────────────────────────────────────────────────────────────
run_clicked = st.button(
    "▶  Lancer le calcul", type="primary", use_container_width=True
)

if run_clicked:

    # Input validation
    if T_min >= T_max:
        st.error("⛔ La température minimale doit être strictement inférieure à la maximale.")
        st.stop()

    params, BDD = load_resources()

    with tempfile.TemporaryDirectory() as tmpdir:

        # ── Solvents ──
        if solvent_file is not None:
            solvent_content = solvent_file.read().decode("utf-8")
        else:
            solvent_content = "smiles zi\nCCCCO 1.0\n"
        solvent_path = os.path.join(tmpdir, "solvents.txt")
        with open(solvent_path, "w", encoding="utf-8") as fh:
            fh.write(solvent_content)

        # ── Solutes ──
        if solute_file is not None:
            solute_content = solute_file.read().decode("utf-8")
        else:
            with open("inputs/solutes.txt", "r", encoding="utf-8") as fh:
                solute_content = fh.read()
        solute_path = os.path.join(tmpdir, "solutes.txt")
        with open(solute_path, "w", encoding="utf-8") as fh:
            fh.write(solute_content)

        # ── Conditions ──
        conditions_content = (
            "Please, inform system conditions:\n"
            f"Pressure[bar] = {P_bar}\n"
            f"Temperature_min[K] = {T_min}\n"
            f"Temperature_max[K] = {T_max}\n"
            f"Dataset = {int(n_T)}\n"
        )
        conditions_path = os.path.join(tmpdir, "system.txt")
        with open(conditions_path, "w", encoding="utf-8") as fh:
            fh.write(conditions_content)

        # ── Load molecules ──
        with st.spinner("Chargement et identification des molécules…"):
            Molecules, zi, Tk, Pbar_val = loadmolecules(
                solvent_path, solute_path, conditions_path, BDD, params
            )
            zi = zi / sum(zi)

        n_mol        = len(Molecules)
        n_T_pts      = len(Tk)
        dataset_size = n_mol * n_T_pts * 3

        st.info(
            f"**{n_mol} molécules** chargées  \n"
            f"Plage de température : {float(Tk.min()):.2f} K → {float(Tk.max()):.2f} K  \n"
            f"Taille du jeu de données : **{dataset_size}** énergies libres de solvatation"
        )

        # ── Compute NASA polynomials ──
        with st.spinner(
            f"Calcul en cours… ({n_T_pts} points × {n_mol} molécules — "
            f"peut prendre {max(10, dataset_size // 200)} s)"
        ):
            t0 = time.time()
            thermochem = fit_nasa(Tk, Pbar_val, zi, Molecules, params)
            elapsed = time.time() - t0

        # ── Build results DataFrame ──
        rows = []
        for i, mol in enumerate(Molecules):
            coeffs = thermochem["NASA_coefficients"][i]
            rows.append(
                {
                    "SMILES":      mol.smiles,
                    "Groupes":     mol.groups,
                    "Composition": float(zi[i, 0]),
                    "Tmin (K)":   float(thermochem["Tmin"]),
                    "Tmax (K)":   float(thermochem["Tmax"]),
                    "R²":         float(thermochem["R2_nasa"][i, 0]),
                    "a1": float(coeffs[0]), "a2": float(coeffs[1]), "a3": float(coeffs[2]),
                    "a4": float(coeffs[3]), "a5": float(coeffs[4]), "a6": float(coeffs[5]), "a7": float(coeffs[6]),
                }
            )
        df = pd.DataFrame(rows)
        st.session_state["results_df"] = df
        st.session_state["elapsed"]    = elapsed

    gc.collect()

# ──────────────────────────────────────────────────────────────
# Results (persistent across re-runs / download clicks)
# ──────────────────────────────────────────────────────────────
if "results_df" in st.session_state:
    df = st.session_state["results_df"]

    st.success(f"✅ Calcul terminé en **{st.session_state['elapsed']:.1f} secondes**")
    st.subheader("📊 Résultats — Coefficients NASA de solvatation")
    fmt = {"Composition": "{:.4f}", "R²": "{:.4f}"}
    fmt.update({f"a{k}": "{:.4e}" for k in range(1, 8)})
    st.dataframe(df.style.format(fmt), use_container_width=True, height=420)

    st.subheader("⬇️ Télécharger les résultats")
    csv_buf = io.StringIO()
    df.to_csv(csv_buf, sep=";", index=False)
    st.download_button(
        label="📄 Télécharger CSV",
        data=csv_buf.getvalue().encode("utf-8"),
        file_name="solv-thermochem.csv",
        mime="text/csv",
    )

# ──────────────────────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "PR/COSMO-RS · BP-TZVPD-FINE · Molécules C/H/O uniquement · "
    "LRGP UMR 7274 CNRS — Université de Lorraine"
)
