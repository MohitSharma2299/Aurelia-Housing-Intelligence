from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Aurelia — Housing Intelligence",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# DESIGN SYSTEM
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
        --bg: #05070b;
        --panel: rgba(13, 17, 25, .70);
        --panel-2: rgba(18, 23, 33, .82);
        --line: rgba(255,255,255,.095);
        --line-strong: rgba(255,255,255,.18);
        --text: #f6f7fb;
        --muted: #8c95a6;
        --muted-2: #667082;
        --violet: #9b8cff;
        --mint: #80f0d0;
        --amber: #ffc978;
        --red: #ff8b9d;
    }

    html, body, [class*="css"] {
        font-family: "Manrope", system-ui, sans-serif;
    }

    .stApp {
        background:
            radial-gradient(900px 500px at 12% -10%, rgba(117, 91, 255, .13), transparent 60%),
            radial-gradient(700px 480px at 94% 8%, rgba(52, 240, 192, .085), transparent 60%),
            radial-gradient(900px 700px at 52% 106%, rgba(255, 163, 96, .05), transparent 62%),
            var(--bg);
        color: var(--text);
    }

    .stApp::before {
        content: "";
        position: fixed;
        inset: 0;
        pointer-events: none;
        z-index: 0;
        opacity: .22;
        background-image:
            linear-gradient(rgba(255,255,255,.018) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255,255,255,.018) 1px, transparent 1px);
        background-size: 46px 46px;
        mask-image: linear-gradient(to bottom, black 0%, transparent 72%);
    }

    .block-container {
        position: relative;
        z-index: 1;
        max-width: 1440px;
        padding: 1.1rem 2rem 4rem;
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stToolbar"] {
        display: none;
    }

    /* ---------------- TOP BAR ---------------- */

    .topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        padding: .35rem 0 1.7rem;
    }

    .brand {
        display: flex;
        align-items: center;
        gap: .75rem;
        font-weight: 800;
        letter-spacing: -.02em;
        color: #fff;
    }

    .brand-mark {
        width: 34px;
        height: 34px;
        display: grid;
        place-items: center;
        border-radius: 12px;
        background: linear-gradient(145deg, rgba(155,140,255,.25), rgba(128,240,208,.10));
        border: 1px solid rgba(255,255,255,.12);
        box-shadow: inset 0 1px rgba(255,255,255,.07), 0 8px 32px rgba(0,0,0,.28);
    }

    .brand-mark::after {
        content: "✦";
        color: #d7d2ff;
        font-size: 15px;
    }

    .top-status {
        display: flex;
        align-items: center;
        gap: .55rem;
        border: 1px solid rgba(128,240,208,.16);
        background: rgba(128,240,208,.045);
        color: #a7f2df;
        padding: .5rem .75rem;
        border-radius: 999px;
        font: 500 .68rem "DM Mono", monospace;
        letter-spacing: .08em;
    }

    .pulse {
        position: relative;
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: var(--mint);
        box-shadow: 0 0 15px rgba(128,240,208,.8);
    }

    .pulse::before {
        content: "";
        position: absolute;
        inset: -5px;
        border: 1px solid rgba(128,240,208,.30);
        border-radius: 50%;
        animation: pulse 1.9s ease-out infinite;
    }

    @keyframes pulse {
        0% { transform: scale(.55); opacity: .9; }
        100% { transform: scale(1.8); opacity: 0; }
    }

    /* ---------------- HERO ---------------- */

    .hero-shell {
        min-height: 410px;
        position: relative;
        overflow: hidden;
        border: 1px solid var(--line);
        border-radius: 34px;
        background:
            linear-gradient(145deg, rgba(255,255,255,.055), rgba(255,255,255,.015));
        box-shadow:
            0 40px 120px rgba(0,0,0,.34),
            inset 0 1px rgba(255,255,255,.045);
        backdrop-filter: blur(20px);
    }

    .hero-copy {
        position: relative;
        z-index: 3;
        padding: 3.6rem 3.5rem;
        max-width: 720px;
    }

    .kicker {
        color: #a4adbc;
        font: 500 .68rem "DM Mono", monospace;
        letter-spacing: .16em;
        text-transform: uppercase;
        margin-bottom: 1rem;
    }

    .hero-title {
        margin: 0;
        font-size: clamp(3.3rem, 7vw, 6.8rem);
        line-height: .9;
        letter-spacing: -.075em;
        font-weight: 800;
        color: #fff;
    }

    .hero-title .accent {
        color: transparent;
        background: linear-gradient(100deg, #fff 12%, #c8c1ff 48%, #8bf2db 92%);
        background-clip: text;
        -webkit-background-clip: text;
    }

    .hero-sub {
        max-width: 620px;
        color: #939cad;
        line-height: 1.72;
        font-size: .98rem;
        margin: 1.35rem 0 1.75rem;
    }

    .hero-meta {
        display: flex;
        flex-wrap: wrap;
        gap: .55rem;
    }

    .hero-chip {
        display: inline-flex;
        align-items: center;
        gap: .4rem;
        padding: .52rem .7rem;
        border-radius: 999px;
        border: 1px solid rgba(255,255,255,.08);
        background: rgba(255,255,255,.025);
        color: #b1b8c5;
        font-size: .7rem;
        font-weight: 700;
    }

    .hero-visual {
        position: absolute;
        inset: 0 0 0 49%;
    }

    .orbit {
        position: absolute;
        width: 330px;
        height: 330px;
        top: 35px;
        right: 11%;
        border: 1px solid rgba(155,140,255,.14);
        border-radius: 50%;
        box-shadow:
            inset 0 0 80px rgba(155,140,255,.03),
            0 0 80px rgba(155,140,255,.04);
        animation: orbitFloat 8s ease-in-out infinite;
    }

    .orbit::before,
    .orbit::after {
        content: "";
        position: absolute;
        border-radius: 50%;
        border: 1px solid rgba(128,240,208,.10);
        inset: 26px;
    }

    .orbit::after {
        inset: 72px;
        border-color: rgba(255,201,120,.09);
    }

    @keyframes orbitFloat {
        0%, 100% { transform: translate3d(0,0,0) rotate(0deg); }
        50% { transform: translate3d(0,-12px,0) rotate(5deg); }
    }

    .orbit-core {
        position: absolute;
        width: 106px;
        height: 106px;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        border-radius: 34px;
        display: grid;
        place-items: center;
        color: #f7f8fc;
        font-size: 34px;
        background:
            radial-gradient(circle at 35% 30%, rgba(255,255,255,.12), transparent 48%),
            linear-gradient(145deg, rgba(155,140,255,.18), rgba(128,240,208,.08));
        border: 1px solid rgba(255,255,255,.14);
        box-shadow:
            0 25px 75px rgba(0,0,0,.45),
            inset 0 1px rgba(255,255,255,.1);
        animation: corePulse 4.8s ease-in-out infinite;
    }

    @keyframes corePulse {
        0%, 100% { box-shadow: 0 25px 75px rgba(0,0,0,.45), 0 0 0 rgba(155,140,255,0); }
        50% { box-shadow: 0 25px 90px rgba(0,0,0,.5), 0 0 55px rgba(155,140,255,.13); }
    }

    .data-node {
        position: absolute;
        border: 1px solid rgba(255,255,255,.12);
        background: rgba(10,14,21,.72);
        backdrop-filter: blur(14px);
        border-radius: 15px;
        padding: .65rem .78rem;
        box-shadow: 0 15px 45px rgba(0,0,0,.25);
        font: 500 .62rem "DM Mono", monospace;
        color: #a9b0bf;
    }

    .node-1 { top: 18%; right: 4%; animation: float1 5.5s ease-in-out infinite; }
    .node-2 { bottom: 18%; right: 22%; animation: float2 6s ease-in-out infinite; }
    .node-3 { top: 38%; left: 11%; animation: float3 6.6s ease-in-out infinite; }

    @keyframes float1 { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-10px)} }
    @keyframes float2 { 0%,100%{transform:translateY(0)} 50%{transform:translateY(12px)} }
    @keyframes float3 { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-7px)} }

    .scanline {
        position: absolute;
        top: 0;
        left: 0;
        width: 45%;
        height: 100%;
        opacity: .17;
        background: repeating-linear-gradient(
            to bottom,
            transparent 0px,
            transparent 6px,
            rgba(255,255,255,.032) 7px
        );
        animation: scan 7s linear infinite;
        pointer-events: none;
    }

    @keyframes scan {
        from { transform: translateY(-35px); }
        to { transform: translateY(35px); }
    }

    /* ---------------- GENERIC CARDS ---------------- */

    .section-head {
        display: flex;
        align-items: end;
        justify-content: space-between;
        gap: 1rem;
        margin: 3.1rem 0 1.05rem;
    }

    .section-eyebrow {
        color: #768092;
        font: 500 .64rem "DM Mono", monospace;
        letter-spacing: .15em;
        text-transform: uppercase;
        margin-bottom: .35rem;
    }

    .section-title {
        margin: 0;
        color: #f5f7fb;
        font-size: 1.65rem;
        letter-spacing: -.04em;
        font-weight: 800;
    }

    .section-desc {
        color: #7f899a;
        margin: .3rem 0 0;
        line-height: 1.5;
        font-size: .83rem;
    }

    .glass {
        position: relative;
        overflow: hidden;
        border: 1px solid var(--line);
        border-radius: 24px;
        background: linear-gradient(145deg, rgba(255,255,255,.048), rgba(255,255,255,.012));
        box-shadow:
            0 24px 70px rgba(0,0,0,.25),
            inset 0 1px rgba(255,255,255,.035);
        backdrop-filter: blur(18px);
        transition:
            transform .35s cubic-bezier(.22,1,.36,1),
            border-color .35s ease,
            box-shadow .35s ease,
            background .35s ease;
    }

    .glass:hover {
        transform: translateY(-7px) scale(1.006);
        border-color: rgba(255,255,255,.17);
        box-shadow:
            0 36px 100px rgba(0,0,0,.34),
            0 0 0 1px rgba(155,140,255,.035),
            inset 0 1px rgba(255,255,255,.07);
        background: linear-gradient(145deg, rgba(255,255,255,.062), rgba(255,255,255,.018));
    }

    .glass::before {
        content: "";
        position: absolute;
        inset: 0;
        pointer-events: none;
        opacity: 0;
        transition: opacity .35s ease;
        background: radial-gradient(
            420px circle at var(--mx, 50%) var(--my, 50%),
            rgba(155,140,255,.10),
            transparent 48%
        );
    }

    .glass:hover::before { opacity: 1; }

    .card-pad { padding: 1.3rem; }

    .micro {
        font: 500 .6rem "DM Mono", monospace;
        text-transform: uppercase;
        letter-spacing: .12em;
        color: #6f798a;
    }

    .value-xl {
        margin-top: .55rem;
        font-size: 1.7rem;
        font-weight: 800;
        letter-spacing: -.05em;
        color: #fff;
    }

    /* ---------------- PREDICTION LAB ---------------- */

    [data-testid="stForm"] {
        position: relative;
        overflow: hidden;
        padding: 1.35rem 1.35rem 1.25rem;
        border-radius: 26px;
        border: 1px solid rgba(255,255,255,.11);
        background:
            radial-gradient(500px 260px at 0% 0%, rgba(155,140,255,.055), transparent 68%),
            linear-gradient(145deg, rgba(255,255,255,.045), rgba(255,255,255,.012));
        box-shadow:
            0 30px 90px rgba(0,0,0,.30),
            inset 0 1px rgba(255,255,255,.045);
        transition: transform .4s cubic-bezier(.22,1,.36,1), border-color .35s ease, box-shadow .35s ease;
    }

    [data-testid="stForm"]:hover {
        transform: translateY(-5px);
        border-color: rgba(255,255,255,.16);
        box-shadow:
            0 38px 110px rgba(0,0,0,.36),
            0 0 60px rgba(155,140,255,.035),
            inset 0 1px rgba(255,255,255,.06);
    }

    [data-testid="stForm"]::before {
        content: "";
        position: absolute;
        inset: 0;
        pointer-events: none;
        opacity: .85;
        background: radial-gradient(
            460px circle at var(--mx, 50%) var(--my, 20%),
            rgba(155,140,255,.055),
            transparent 52%
        );
    }


    .lab {
        position: relative;
        overflow: hidden;
        padding: 1.35rem;
        border-radius: 28px;
        border: 1px solid rgba(255,255,255,.11);
        background:
            radial-gradient(500px 250px at 0% 0%, rgba(155,140,255,.06), transparent 65%),
            linear-gradient(145deg, rgba(255,255,255,.052), rgba(255,255,255,.012));
        box-shadow:
            0 30px 95px rgba(0,0,0,.34),
            inset 0 1px rgba(255,255,255,.05);
        transition: transform .4s cubic-bezier(.22,1,.36,1), border-color .35s ease;
    }

    .lab:hover {
        transform: translateY(-5px);
        border-color: rgba(255,255,255,.16);
    }

    .lab-head {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        padding: .25rem .35rem 1.15rem;
    }

    .lab-title {
        font-size: 1.1rem;
        font-weight: 800;
        letter-spacing: -.025em;
        color: #fff;
    }

    .lab-copy {
        margin-top: .2rem;
        color: #788294;
        font-size: .75rem;
    }

    .lab-badge {
        color: #c8cfff;
        border: 1px solid rgba(155,140,255,.17);
        background: rgba(155,140,255,.055);
        padding: .45rem .65rem;
        border-radius: 999px;
        font: 500 .6rem "DM Mono", monospace;
        letter-spacing: .08em;
        white-space: nowrap;
    }

    /* Streamlit native input treatment */

    div[data-testid="stNumberInput"],
    div[data-testid="stSlider"],
    div[data-testid="stButton"] {
        transition: transform .25s ease;
    }

    div[data-testid="stNumberInput"]:hover {
        transform: translateY(-2px);
    }

    div[data-testid="stNumberInput"] label,
    div[data-testid="stSlider"] label {
        color: #aeb6c4 !important;
        font-size: .74rem !important;
        font-weight: 700 !important;
    }

    div[data-baseweb="input"] {
        background: rgba(255,255,255,.025) !important;
        border: 1px solid rgba(255,255,255,.08) !important;
        border-radius: 13px !important;
        min-height: 45px !important;
        transition: border-color .25s ease, box-shadow .25s ease, background .25s ease;
    }

    div[data-baseweb="input"]:hover,
    div[data-baseweb="input"]:focus-within {
        border-color: rgba(155,140,255,.34) !important;
        background: rgba(155,140,255,.035) !important;
        box-shadow: 0 0 0 1px rgba(155,140,255,.09), 0 10px 30px rgba(0,0,0,.14) !important;
    }

    input {
        color: #f7f8fb !important;
        font-weight: 600 !important;
    }

    button[aria-label^="Increment"],
    button[aria-label^="Decrement"] {
        background: transparent !important;
        color: #707a8c !important;
    }

    div[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
        box-shadow: 0 0 20px rgba(155,140,255,.18);
    }

    div[data-testid="stFormSubmitButton"] button {
        width: 100% !important;
        border: 1px solid rgba(255,255,255,.18) !important;
        border-radius: 15px !important;
        padding: .78rem 1rem !important;
        color: #080b10 !important;
        font-weight: 800 !important;
        background: linear-gradient(135deg, #c8c2ff, #9cf1dd) !important;
        box-shadow: 0 15px 45px rgba(128,240,208,.08) !important;
        transition: transform .3s cubic-bezier(.22,1,.36,1), box-shadow .3s ease, filter .3s ease !important;
    }

    div[data-testid="stFormSubmitButton"] button:hover {
        transform: translateY(-4px) scale(1.015) !important;
        box-shadow: 0 22px 65px rgba(128,240,208,.15) !important;
        filter: brightness(1.04);
    }

    .reset-button > div > button {
        background: rgba(255,255,255,.025) !important;
        color: #a7afbd !important;
        border: 1px solid rgba(255,255,255,.08) !important;
        border-radius: 15px !important;
    }

    /* ---------------- RESULT ---------------- */

    .result-shell {
        min-height: 470px;
        position: relative;
        overflow: hidden;
        border-radius: 28px;
        border: 1px solid rgba(255,255,255,.11);
        padding: 1.55rem;
        background:
            radial-gradient(450px 270px at 85% 8%, rgba(155,140,255,.13), transparent 62%),
            radial-gradient(400px 240px at 8% 92%, rgba(128,240,208,.08), transparent 65%),
            linear-gradient(145deg, rgba(255,255,255,.050), rgba(255,255,255,.012));
        box-shadow:
            0 35px 100px rgba(0,0,0,.34),
            inset 0 1px rgba(255,255,255,.045);
    }

    .result-shell::after {
        content: "";
        position: absolute;
        width: 260px;
        height: 260px;
        right: -120px;
        top: -140px;
        border-radius: 50%;
        border: 1px solid rgba(155,140,255,.10);
        box-shadow:
            0 0 0 28px rgba(155,140,255,.018),
            0 0 0 56px rgba(155,140,255,.012);
        animation: resultOrbit 9s linear infinite;
        pointer-events: none;
    }

    @keyframes resultOrbit {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }

    .result-top {
        display: flex;
        justify-content: space-between;
        gap: 1rem;
        align-items: flex-start;
    }

    .result-label {
        color: #7c8697;
        font: 500 .64rem "DM Mono", monospace;
        text-transform: uppercase;
        letter-spacing: .14em;
    }

    .result-tag {
        padding: .4rem .6rem;
        border-radius: 999px;
        border: 1px solid rgba(128,240,208,.15);
        background: rgba(128,240,208,.04);
        color: #9cf0db;
        font: 500 .58rem "DM Mono", monospace;
    }

    .result-number {
        margin: 1.15rem 0 .35rem;
        font-size: clamp(3.5rem, 6vw, 5.7rem);
        line-height: .88;
        font-weight: 800;
        letter-spacing: -.075em;
        color: #fff;
        text-shadow: 0 0 55px rgba(155,140,255,.08);
    }

    .result-caption {
        color: #80899a;
        font-size: .76rem;
        line-height: 1.55;
    }

    .result-divider {
        height: 1px;
        margin: 1.6rem 0;
        background: linear-gradient(90deg, rgba(255,255,255,.09), transparent);
    }

    .result-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: .7rem;
        position: relative;
        z-index: 2;
    }

    .metric {
        border-radius: 15px;
        border: 1px solid rgba(255,255,255,.07);
        background: rgba(255,255,255,.02);
        padding: .85rem;
        transition: transform .28s ease, border-color .28s ease, background .28s ease;
    }

    .metric:hover {
        transform: translateY(-4px);
        border-color: rgba(255,255,255,.14);
        background: rgba(255,255,255,.035);
    }

    .metric-label {
        color: #677182;
        font: 500 .58rem "DM Mono", monospace;
        text-transform: uppercase;
        letter-spacing: .11em;
    }

    .metric-value {
        margin-top: .35rem;
        color: #f6f7fb;
        font-size: 1.05rem;
        font-weight: 800;
    }

    /* ---------------- CHIP ROW ---------------- */

    .insight {
        display: flex;
        align-items: center;
        gap: .65rem;
        padding: .72rem .8rem;
        margin-top: .7rem;
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,.07);
        background: rgba(255,255,255,.018);
        color: #9aa4b4;
        font-size: .69rem;
        line-height: 1.45;
    }

    .insight-icon {
        width: 25px;
        height: 25px;
        flex: 0 0 25px;
        display: grid;
        place-items: center;
        border-radius: 8px;
        color: #c9c2ff;
        background: rgba(155,140,255,.08);
        border: 1px solid rgba(155,140,255,.12);
        font-size: 11px;
    }

    /* ---------------- CHART WRAPPERS ---------------- */

    [data-testid="stPlotlyChart"] {
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,.075);
        border-radius: 22px;
        background:
            radial-gradient(430px 220px at 8% 0%, rgba(155,140,255,.035), transparent 62%),
            linear-gradient(145deg, rgba(255,255,255,.032), rgba(255,255,255,.012));
        box-shadow:
            0 22px 70px rgba(0,0,0,.22),
            inset 0 1px rgba(255,255,255,.035);
        transition:
            transform .35s cubic-bezier(.22,1,.36,1),
            border-color .35s ease,
            box-shadow .35s ease;
    }

    [data-testid="stPlotlyChart"]:hover {
        transform: translateY(-5px);
        border-color: rgba(255,255,255,.15);
        box-shadow:
            0 30px 90px rgba(0,0,0,.30),
            0 0 40px rgba(155,140,255,.045);
    }

    [data-testid="stPlotlyChart"] .js-plotly-plot,
    [data-testid="stPlotlyChart"] .plot-container {
        border-radius: 22px;
    }

    .chart-toolbar-note {
        display:flex;
        justify-content:flex-end;
        color:#646e7e;
        font:500 .59rem "DM Mono", monospace;
        letter-spacing:.08em;
        text-transform:uppercase;
        margin:-.35rem 0 .55rem;
    }


    .chart-wrap {
        padding: .55rem .75rem .2rem;
        border: 1px solid rgba(255,255,255,.075);
        border-radius: 21px;
        background: rgba(255,255,255,.018);
        transition: transform .35s cubic-bezier(.22,1,.36,1), border-color .35s ease;
    }

    .chart-wrap:hover {
        transform: translateY(-5px);
        border-color: rgba(255,255,255,.14);
    }

    /* ---------------- EXPLAINER ---------------- */

    .pipeline {
        display: grid;
        grid-template-columns: 1fr 80px 1fr 80px 1fr;
        align-items: center;
        gap: .7rem;
        padding: 1.2rem;
        border: 1px solid rgba(255,255,255,.075);
        border-radius: 24px;
        background:
            linear-gradient(145deg, rgba(255,255,255,.042), rgba(255,255,255,.01));
    }

    .pipeline-node {
        min-height: 145px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        border: 1px solid rgba(255,255,255,.075);
        border-radius: 19px;
        padding: 1rem;
        background: rgba(255,255,255,.018);
        transition: transform .3s ease, border-color .3s ease, background .3s ease;
    }

    .pipeline-node:hover {
        transform: translateY(-6px);
        border-color: rgba(155,140,255,.22);
        background: rgba(155,140,255,.025);
    }

    .node-title {
        color: #fff;
        font-weight: 800;
        margin-top: .35rem;
    }

    .node-copy {
        color: #747f90;
        font-size: .69rem;
        line-height: 1.55;
    }

    .arrow {
        text-align: center;
        color: #666f80;
        font-size: 1.4rem;
    }

    /* ---------------- FOOTER ---------------- */

    .footer {
        margin-top: 4rem;
        padding-top: 1.5rem;
        border-top: 1px solid rgba(255,255,255,.06);
        color: #5f6878;
        font: 500 .62rem "DM Mono", monospace;
        display: flex;
        justify-content: space-between;
        gap: 1rem;
        flex-wrap: wrap;
    }

    /* ---------------- MOBILE ---------------- */

    @media (max-width: 900px) {
        .block-container { padding: .8rem 1rem 3rem; }
        .hero-shell { min-height: 540px; }
        .hero-copy { padding: 2.3rem 1.6rem; }
        .hero-visual { inset: 48% 0 0 0; opacity: .7; }
        .orbit { width: 270px; height: 270px; right: 8%; top: 25px; }
        .pipeline { grid-template-columns: 1fr; }
        .arrow { transform: rotate(90deg); }
    }

    @media (max-width: 600px) {
        .topbar { align-items: flex-start; }
        .top-status { font-size: .55rem; }
        .hero-title { font-size: 3.1rem; }
        .result-number { font-size: 3.8rem; }
        .result-grid { grid-template-columns: 1fr; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD MODEL + DATA
# ============================================================

MODEL_PATH = Path(__file__).resolve().parent / "lr_cal_prac.pkl"


@st.cache_resource
def load_model(model_path: str):
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    return bundle["model"], list(bundle["columns"])


@st.cache_data
def load_housing_data():
    frame = fetch_california_housing(as_frame=True).frame
    return frame


if not MODEL_PATH.exists():
    st.error(
        f"Could not find `{MODEL_PATH.name}`. Put the trained pickle file "
        "in the same folder as this app."
    )
    st.stop()

try:
    model, feature_columns = load_model(str(MODEL_PATH))
except Exception as exc:
    st.error(f"Model loading failed: {exc}")
    st.stop()

try:
    housing = load_housing_data()
except Exception:
    housing = None


# ============================================================
# METRICS — COMPUTED, NEVER HARD-CODED
# ============================================================

metrics = {}

if housing is not None:
    X = housing[feature_columns]
    y = housing["MedHouseVal"]

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=1234,
    )

    test_pred = model.predict(X_test)

    metrics["RMSE"] = root_mean_squared_error(y_test, test_pred)
    metrics["MSE"] = mean_squared_error(y_test, test_pred)
    metrics["R2"] = r2_score(y_test, test_pred)

    residuals = y_test - test_pred

else:
    residuals = pd.Series(dtype=float)


# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "prediction": None,
    "prediction_inputs": None,
    "history": [],
    "scenario_feature": "MedInc",
    "scenario_delta": 0.0,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# TOP BAR
# ============================================================

st.markdown(
    """
    <div class="topbar">
        <div class="brand">
            <div class="brand-mark"></div>
            <div>Aurelia / Housing Intelligence</div>
        </div>

        <div class="top-status">
            <span class="pulse"></span>
            MODEL ONLINE · LINEAR REGRESSION
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HERO
# ============================================================

components.html(
    """
    <div id="aurelia-hero"
         style="
            height: 410px;
            width: 100%;
            overflow: hidden;
            border-radius: 34px;
            position: relative;
            font-family: Manrope, system-ui, sans-serif;
         ">
    </div>

    <script type="module">
      import React from "https://esm.sh/react@18.3.1";
      import { createRoot } from "https://esm.sh/react-dom@18.3.1/client";
      import { motion } from "https://esm.sh/framer-motion@11.18.0?deps=react@18.3.1,react-dom@18.3.1";

      const h = React.createElement;

      function App() {
        return h(
          motion.div,
          {
            initial: { opacity: 0, y: 24 },
            animate: { opacity: 1, y: 0 },
            transition: { duration: 0.8, ease: [0.22, 1, 0.36, 1] },
            style: {
              width: "100%",
              height: "100%",
              position: "relative",
              overflow: "hidden",
              border: "1px solid rgba(255,255,255,.10)",
              borderRadius: "34px",
              background:
                "radial-gradient(600px 420px at 78% 46%, rgba(110,92,255,.12), transparent 62%), radial-gradient(500px 340px at 90% 85%, rgba(55,238,198,.075), transparent 65%), linear-gradient(145deg, rgba(255,255,255,.055), rgba(255,255,255,.012))",
              boxShadow: "0 40px 120px rgba(0,0,0,.34), inset 0 1px rgba(255,255,255,.05)"
            }
          },

          h(
            motion.div,
            {
              animate: { rotate: 360 },
              transition: { duration: 30, repeat: Infinity, ease: "linear" },
              style: {
                position: "absolute",
                width: "390px",
                height: "390px",
                right: "8%",
                top: "10px",
                borderRadius: "50%",
                border: "1px solid rgba(155,140,255,.16)",
                boxShadow: "inset 0 0 80px rgba(155,140,255,.035)"
              }
            },

            h("div", {
              style: {
                position: "absolute",
                inset: "42px",
                borderRadius: "50%",
                border: "1px solid rgba(128,240,208,.10)"
              }
            }),

            h("div", {
              style: {
                position: "absolute",
                inset: "95px",
                borderRadius: "50%",
                border: "1px solid rgba(255,201,120,.10)"
              }
            }),

            h(
              motion.div,
              {
                animate: {
                  y: [0, -12, 0],
                  rotateZ: [-2, 1.5, -2],
                  rotateX: [0, 3, 0]
                },
                transition: {
                  duration: 5.2,
                  repeat: Infinity,
                  ease: "easeInOut"
                },
                whileHover: {
                  scale: 1.08,
                  rotateZ: 0,
                  transition: { duration: .25 }
                },
                style: {
                  position: "absolute",
                  width: "132px",
                  height: "112px",
                  left: "50%",
                  top: "50%",
                  transform: "translate(-50%,-50%)",
                  display: "grid",
                  placeItems: "center",
                  borderRadius: "31px",
                  background:
                    "radial-gradient(circle at 30% 18%, rgba(255,255,255,.15), transparent 42%), linear-gradient(145deg, rgba(155,140,255,.20), rgba(128,240,208,.10))",
                  border: "1px solid rgba(255,255,255,.17)",
                  boxShadow:
                    "0 28px 90px rgba(0,0,0,.48), 0 0 70px rgba(155,140,255,.11), inset 0 1px rgba(255,255,255,.09)"
                }
              },
              h("div", {
                dangerouslySetInnerHTML: {
                  __html: `
                    <svg width="92" height="82" viewBox="0 0 92 82" fill="none"
                         xmlns="http://www.w3.org/2000/svg"
                         style="filter: drop-shadow(0 10px 22px rgba(0,0,0,.32)); overflow:visible;">
                      <defs>
                        <linearGradient id="roof" x1="18" y1="18" x2="76" y2="63" gradientUnits="userSpaceOnUse">
                          <stop stop-color="#D9D2FF"/>
                          <stop offset="1" stop-color="#9A8AFF"/>
                        </linearGradient>
                        <linearGradient id="wall" x1="28" y1="35" x2="66" y2="73" gradientUnits="userSpaceOnUse">
                          <stop stop-color="#F5F7FB"/>
                          <stop offset="1" stop-color="#BFC6D6"/>
                        </linearGradient>
                        <linearGradient id="window" x1="0" y1="0" x2="1" y2="1">
                          <stop stop-color="#A5F2E1"/>
                          <stop offset="1" stop-color="#7A8CFF"/>
                        </linearGradient>
                      </defs>

                      <ellipse cx="46" cy="76" rx="23" ry="4"
                               fill="rgba(128,240,208,.18)"/>

                      <path d="M14 37L46 10L78 37V66C78 69.314 75.314 72 72 72H20C16.686 72 14 69.314 14 66V37Z"
                            fill="url(#wall)"
                            stroke="rgba(255,255,255,.7)"
                            stroke-width="1.2"/>

                      <path d="M8 39L46 7L84 39L78 45L46 18L14 45L8 39Z"
                            fill="url(#roof)"
                            stroke="rgba(255,255,255,.35)"
                            stroke-width="1.2"/>

                      <path d="M39 72V51C39 48.239 41.239 46 44 46H48C50.761 46 53 48.239 53 51V72"
                            fill="#121824"
                            stroke="rgba(120,130,150,.42)"
                            stroke-width="1"/>

                      <rect x="22" y="48" width="11" height="11" rx="2"
                            fill="url(#window)"
                            stroke="rgba(255,255,255,.55)"
                            stroke-width="1"/>

                      <path d="M27.5 48V59M22 53.5H33"
                            stroke="rgba(5,10,16,.48)"
                            stroke-width="1"/>

                      <rect x="59" y="48" width="11" height="11" rx="2"
                            fill="url(#window)"
                            stroke="rgba(255,255,255,.55)"
                            stroke-width="1"/>

                      <path d="M64.5 48V59M59 53.5H70"
                            stroke="rgba(5,10,16,.48)"
                            stroke-width="1"/>

                      <circle cx="71" cy="35" r="2.1" fill="#80F0D0"/>
                    </svg>
                  `
                }
              })
            )
          ),

          h(
            motion.div,
            {
              animate: { y: [0, -10, 0] },
              transition: { duration: 6, repeat: Infinity, ease: "easeInOut" },
              style: {
                position: "absolute",
                right: "4%",
                top: "17%",
                padding: "10px 12px",
                borderRadius: "15px",
                border: "1px solid rgba(255,255,255,.11)",
                background: "rgba(9,13,19,.70)",
                backdropFilter: "blur(14px)",
                color: "#aeb7c6",
                fontFamily: "DM Mono, monospace",
                fontSize: "10px"
              }
            },
            "R²  /  LIVE"
          ),

          h(
            motion.div,
            {
              animate: { y: [0, 11, 0] },
              transition: { duration: 6.5, repeat: Infinity, ease: "easeInOut" },
              style: {
                position: "absolute",
                right: "23%",
                bottom: "17%",
                padding: "10px 12px",
                borderRadius: "15px",
                border: "1px solid rgba(255,255,255,.11)",
                background: "rgba(9,13,19,.70)",
                backdropFilter: "blur(14px)",
                color: "#aeb7c6",
                fontFamily: "DM Mono, monospace",
                fontSize: "10px"
              }
            },
            "8  /  FEATURES"
          ),

          h(
            motion.div,
            {
              animate: { y: [0, -7, 0] },
              transition: { duration: 5.8, repeat: Infinity, ease: "easeInOut" },
              style: {
                position: "absolute",
                left: "52%",
                top: "41%",
                padding: "10px 12px",
                borderRadius: "15px",
                border: "1px solid rgba(255,255,255,.11)",
                background: "rgba(9,13,19,.70)",
                backdropFilter: "blur(14px)",
                color: "#aeb7c6",
                fontFamily: "DM Mono, monospace",
                fontSize: "10px"
              }
            },
            "PREDICT  /  EXPLAIN"
          ),

          h(
            motion.div,
            {
              initial: { opacity: 0, x: -18 },
              animate: { opacity: 1, x: 0 },
              transition: { delay: .15, duration: .7 },
              style: {
                position: "absolute",
                zIndex: 5,
                left: "3.7rem",
                top: "3.45rem",
                maxWidth: "680px"
              }
            },

            h("div", {
              style: {
                color: "#9ba4b3",
                fontFamily: "DM Mono, monospace",
                fontSize: "10px",
                letterSpacing: "2.2px",
                textTransform: "uppercase",
                marginBottom: "15px"
              }
            }, "MACHINE LEARNING / PREDICTIVE SYSTEM"),

            h("h1", {
              style: {
                margin: 0,
                fontSize: "clamp(48px, 7vw, 92px)",
                lineHeight: ".90",
                letterSpacing: "-5.5px",
                fontWeight: 800,
                color: "#fff"
              }
            },
              "Housing",
              h("br"),
              h("span", {
                style: {
                  color: "transparent",
                  background: "linear-gradient(100deg,#fff 10%,#c9c2ff 52%,#8bf0da 94%)",
                  WebkitBackgroundClip: "text",
                  WebkitTextFillColor: "transparent"
                }
              }, "Intelligence")
            ),

            h(
              motion.p,
              {
                initial: { opacity: 0 },
                animate: { opacity: 1 },
                transition: { delay: .42, duration: .65 },
                style: {
                  margin: "20px 0 22px",
                  color: "#929baa",
                  fontSize: "14px",
                  lineHeight: 1.68,
                  maxWidth: "580px"
                }
              },
              "Explore a trained linear model as an interactive system — from property signals to prediction, model behavior, and explanation."
            ),

            h("div", {
              style: {
                display: "flex",
                flexWrap: "wrap",
                gap: "7px"
              }
            },
              ...["LIVE INFERENCE", "MODEL INTERPRETABILITY", "SCENARIO LAB"].map(
                (item) =>
                  h("div", {
                    key: item,
                    style: {
                      padding: "8px 11px",
                      border: "1px solid rgba(255,255,255,.08)",
                      borderRadius: "999px",
                      background: "rgba(255,255,255,.025)",
                      color: "#aeb6c4",
                      fontSize: "10px",
                      fontWeight: 700
                    }
                  }, item)
              )
            )
          ),

          h("div", {
            style: {
              position: "absolute",
              left: 0,
              top: 0,
              width: "48%",
              height: "100%",
              opacity: .12,
              background:
                "repeating-linear-gradient(to bottom,transparent 0,transparent 6px,rgba(255,255,255,.04) 7px)",
              pointerEvents: "none"
            }
          })
        );
      }

      createRoot(document.getElementById("aurelia-hero")).render(h(App));
    </script>
    """,
    height=425,
    scrolling=False,
)


# ============================================================
# OVERVIEW METRICS
# ============================================================

st.markdown(
    """
    <div class="section-head">
        <div>
            <div class="section-eyebrow">01 / MODEL SNAPSHOT</div>
            <h2 class="section-title">The system at a glance</h2>
            <p class="section-desc">Real values calculated from the saved model and California Housing data.</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

m1, m2, m3, m4 = st.columns(4)

snapshot = [
    ("ALGORITHM", "Linear Regression", "interpretable baseline"),
    ("FEATURES", str(len(feature_columns)), "model inputs"),
    ("RMSE", f"{metrics['RMSE']:.4f}" if metrics else "—", "test-set error"),
    ("R²", f"{metrics['R2']:.4f}" if metrics else "—", "test-set variance explained"),
]

for col, (label, value, desc) in zip([m1, m2, m3, m4], snapshot):
    with col:
        st.markdown(
            f"""
            <div class="glass card-pad">
                <div class="micro">{label}</div>
                <div class="value-xl">{value}</div>
                <div style="margin-top:.35rem;color:#697486;font-size:.67rem;">{desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# PREDICTION LAB
# ============================================================

st.markdown(
    """
    <div class="section-head">
        <div>
            <div class="section-eyebrow">02 / LIVE INFERENCE</div>
            <h2 class="section-title">Prediction laboratory</h2>
            <p class="section-desc">Shape a property profile, run the model, then inspect what changed.</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([1.08, .92], gap="large")


with left:
    st.markdown(
        """
        <div style="display:flex;align-items:center;justify-content:space-between;gap:1rem;margin-bottom:.8rem;">
            <div>
                <div class="micro">INFERENCE MODULE</div>
                <div class="lab-title">Property signal map</div>
                <div class="lab-copy">Every control maps directly to a feature expected by the trained model.</div>
            </div>
            <div class="lab-badge">LIVE</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("inference_form", clear_on_submit=False):
        a, b = st.columns(2)

        with a:
            med_inc = st.number_input(
                "Median income",
                value=5.0,
                step=0.1,
                format="%.2f",
                help="Median income in the block group.",
            )
            house_age = st.number_input(
                "House age",
                value=25.0,
                step=1.0,
                format="%.1f",
                help="Median age of houses in the block group.",
            )
            ave_rooms = st.number_input(
                "Average rooms",
                value=5.5,
                step=0.1,
                format="%.2f",
                help="Average number of rooms per household.",
            )
            ave_bedrms = st.number_input(
                "Average bedrooms",
                value=1.0,
                step=0.1,
                format="%.2f",
                help="Average number of bedrooms per household.",
            )

        with b:
            population = st.number_input(
                "Population",
                value=1500.0,
                step=50.0,
                format="%.0f",
                help="Block-group population.",
            )
            ave_occup = st.number_input(
                "Average occupancy",
                value=3.0,
                step=0.1,
                format="%.2f",
                help="Average number of occupants per household.",
            )
            latitude = st.number_input(
                "Latitude",
                value=34.0,
                step=0.01,
                format="%.4f",
                help="Geographic latitude.",
            )
            longitude = st.number_input(
                "Longitude",
                value=-118.0,
                step=0.01,
                format="%.4f",
                help="Geographic longitude.",
            )

        st.markdown("<div style='height:.45rem'></div>", unsafe_allow_html=True)

        p1, p2 = st.columns([3, 1])

        with p1:
            submitted = st.form_submit_button("Run inference  ↗", use_container_width=True)
            
        with p2:
            reset = st.form_submit_button("Reset", use_container_width=True)
            
    
    if reset:
        st.session_state.prediction = None
        st.session_state.prediction_inputs = None
        st.rerun()

    if submitted:
        values = {
            "MedInc": med_inc,
            "HouseAge": house_age,
            "AveRooms": ave_rooms,
            "AveBedrms": ave_bedrms,
            "Population": population,
            "AveOccup": ave_occup,
            "Latitude": latitude,
            "Longitude": longitude,
        }

        input_df = pd.DataFrame([values])[feature_columns]
        prediction = float(model.predict(input_df)[0])

        st.session_state.prediction = prediction
        st.session_state.prediction_inputs = input_df

        st.session_state.history.append(
            {
                "prediction": prediction,
                **values,
            }
        )
        st.session_state.history = st.session_state.history[-8:]


with right:

    prediction = st.session_state.prediction

    if prediction is None:
        components.html(
            """
            <div class="empty-result" style="
                width:100%; min-height:470px; box-sizing:border-box;
                position:relative; overflow:hidden;
                border-radius:28px;
                border:1px solid rgba(255,255,255,.11);
                padding:28px;
                background:
                    radial-gradient(420px 240px at 84% 8%, rgba(155,140,255,.11), transparent 62%),
                    radial-gradient(360px 220px at 8% 95%, rgba(128,240,208,.07), transparent 65%),
                    linear-gradient(145deg, rgba(255,255,255,.05), rgba(255,255,255,.012));
                box-shadow:0 35px 100px rgba(0,0,0,.34), inset 0 1px rgba(255,255,255,.045);
                font-family:Manrope,system-ui,sans-serif;
                color:#fff;
            ">
                <div style="
                    width:44px;height:44px;border-radius:15px;display:grid;place-items:center;
                    background:linear-gradient(145deg,rgba(155,140,255,.16),rgba(128,240,208,.07));
                    border:1px solid rgba(255,255,255,.10);
                    color:#d7d2ff;font-size:18px;
                    box-shadow:0 14px 35px rgba(0,0,0,.25), inset 0 1px rgba(255,255,255,.07);
                    animation:floatIcon 4.5s ease-in-out infinite;
                ">✦</div>

                <div style="margin-top:24px;color:#7f8999;font:500 10px 'DM Mono',monospace;letter-spacing:.14em;text-transform:uppercase;">
                    READY WHEN YOU ARE
                </div>

                <div style="margin-top:9px;font-size:31px;line-height:1.03;letter-spacing:-.055em;font-weight:800;">
                    Your prediction is waiting.
                </div>

                <div style="margin-top:14px;max-width:390px;color:#8d97a7;font-size:13px;line-height:1.7;">
                    Enter the property details on the left. Then run the model to see the estimated median house value here.
                </div>

                <div style="height:1px;margin:25px 0 18px;background:linear-gradient(90deg,rgba(255,255,255,.10),transparent);"></div>

                <div style="display:grid;gap:9px;">
                    <div style="display:flex;align-items:center;gap:11px;padding:11px 12px;border:1px solid rgba(255,255,255,.065);background:rgba(255,255,255,.018);border-radius:14px;">
                        <div style="width:25px;height:25px;display:grid;place-items:center;border-radius:8px;background:rgba(155,140,255,.08);border:1px solid rgba(155,140,255,.12);color:#cbc4ff;font:700 11px 'DM Mono',monospace;">01</div>
                        <div><div style="font-size:11px;font-weight:800;color:#e9ebf0;">Describe the property</div><div style="font-size:10px;color:#727d8e;margin-top:2px;">Use the eight fields on the left.</div></div>
                    </div>

                    <div style="display:flex;align-items:center;gap:11px;padding:11px 12px;border:1px solid rgba(255,255,255,.065);background:rgba(255,255,255,.018);border-radius:14px;">
                        <div style="width:25px;height:25px;display:grid;place-items:center;border-radius:8px;background:rgba(128,240,208,.07);border:1px solid rgba(128,240,208,.12);color:#a1efdd;font:700 11px 'DM Mono',monospace;">02</div>
                        <div><div style="font-size:11px;font-weight:800;color:#e9ebf0;">Run the model</div><div style="font-size:10px;color:#727d8e;margin-top:2px;">Click <b style="color:#aab4c3;">Run inference</b>.</div></div>
                    </div>

                    <div style="display:flex;align-items:center;gap:11px;padding:11px 12px;border:1px solid rgba(255,255,255,.065);background:rgba(255,255,255,.018);border-radius:14px;">
                        <div style="width:25px;height:25px;display:grid;place-items:center;border-radius:8px;background:rgba(255,201,120,.06);border:1px solid rgba(255,201,120,.12);color:#ffd28b;font:700 11px 'DM Mono',monospace;">03</div>
                        <div><div style="font-size:11px;font-weight:800;color:#e9ebf0;">Read the result</div><div style="font-size:10px;color:#727d8e;margin-top:2px;">Your estimate and model insights appear here.</div></div>
                    </div>
                </div>

                <div style="position:absolute;right:-72px;bottom:-95px;width:220px;height:220px;border-radius:50%;border:1px solid rgba(155,140,255,.08);box-shadow:0 0 0 25px rgba(155,140,255,.014),0 0 0 50px rgba(155,140,255,.010);animation:slowSpin 15s linear infinite;"></div>
            </div>
            <style>
                @keyframes floatIcon { 0%,100%{transform:translateY(0) rotate(0)} 50%{transform:translateY(-6px) rotate(3deg)} }
                @keyframes slowSpin { from{transform:rotate(0deg)} to{transform:rotate(360deg)} }
            </style>
            """,
            height=490,
            scrolling=False,
        )
    else:
        dollar_value = prediction * 100_000
        if dollar_value >= 1_000_000:
            display_value = f"${dollar_value / 1_000_000:.2f}M"
        else:
            display_value = f"${dollar_value:,.0f}"

        components.html(
            f"""
            <div style="
                width:100%; min-height:470px; box-sizing:border-box;
                position:relative; overflow:hidden;
                border-radius:28px;
                border:1px solid rgba(255,255,255,.11);
                padding:27px;
                background:
                    radial-gradient(450px 260px at 86% 7%, rgba(155,140,255,.14), transparent 62%),
                    radial-gradient(380px 220px at 8% 95%, rgba(128,240,208,.08), transparent 65%),
                    linear-gradient(145deg, rgba(255,255,255,.05), rgba(255,255,255,.012));
                box-shadow:0 35px 100px rgba(0,0,0,.34), inset 0 1px rgba(255,255,255,.045);
                font-family:Manrope,system-ui,sans-serif;
                color:#fff;
            ">
                <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:12px;">
                    <div>
                        <div style="color:#7c8696;font:500 9px 'DM Mono',monospace;letter-spacing:.15em;text-transform:uppercase;">PREDICTION COMPLETE</div>
                        <div style="margin-top:8px;font-size:13px;font-weight:700;color:#c7ccd5;">Estimated median house value</div>
                    </div>
                    <div style="padding:6px 8px;border-radius:999px;border:1px solid rgba(128,240,208,.15);background:rgba(128,240,208,.04);color:#9cf0db;font:500 8px 'DM Mono',monospace;">LIVE INFERENCE</div>
                </div>

                <div style="margin-top:27px;font-size:clamp(52px,6vw,82px);line-height:.87;letter-spacing:-.075em;font-weight:800;color:#fff;text-shadow:0 0 55px rgba(155,140,255,.10);">{display_value}</div>

                <div style="margin-top:14px;color:#8993a3;font-size:11px;line-height:1.65;">
                    Estimated median value for the property profile you entered.
                </div>

                <div style="height:1px;margin:22px 0 17px;background:linear-gradient(90deg,rgba(255,255,255,.10),transparent);"></div>

                <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;">
                    <div style="padding:12px;border:1px solid rgba(255,255,255,.065);background:rgba(255,255,255,.018);border-radius:15px;">
                        <div style="color:#697486;font:500 8px 'DM Mono',monospace;letter-spacing:.11em;text-transform:uppercase;">Raw model value</div>
                        <div style="margin-top:5px;font-size:15px;font-weight:800;color:#f3f5f8;">{prediction:.4f}</div>
                    </div>
                    <div style="padding:12px;border:1px solid rgba(255,255,255,.065);background:rgba(255,255,255,.018);border-radius:15px;">
                        <div style="color:#697486;font:500 8px 'DM Mono',monospace;letter-spacing:.11em;text-transform:uppercase;">Model</div>
                        <div style="margin-top:5px;font-size:15px;font-weight:800;color:#f3f5f8;">Linear Regression</div>
                    </div>
                    <div style="padding:12px;border:1px solid rgba(255,255,255,.065);background:rgba(255,255,255,.018);border-radius:15px;">
                        <div style="color:#697486;font:500 8px 'DM Mono',monospace;letter-spacing:.11em;text-transform:uppercase;">Signals used</div>
                        <div style="margin-top:5px;font-size:15px;font-weight:800;color:#f3f5f8;">{len(feature_columns)} features</div>
                    </div>
                    <div style="padding:12px;border:1px solid rgba(255,255,255,.065);background:rgba(255,255,255,.018);border-radius:15px;">
                        <div style="color:#697486;font:500 8px 'DM Mono',monospace;letter-spacing:.11em;text-transform:uppercase;">Target</div>
                        <div style="margin-top:5px;font-size:15px;font-weight:800;color:#f3f5f8;">MedHouseVal</div>
                    </div>
                </div>

                <div style="display:flex;align-items:flex-start;gap:10px;margin-top:13px;padding:12px;border-radius:15px;border:1px solid rgba(128,240,208,.08);background:rgba(128,240,208,.025);color:#8d98a8;font-size:10px;line-height:1.55;">
                    <div style="width:25px;height:25px;flex:0 0 25px;display:grid;place-items:center;border-radius:8px;background:rgba(128,240,208,.07);border:1px solid rgba(128,240,208,.11);color:#9feede;font-weight:800;">✓</div>
                    <div><strong style="color:#dce7e3;">What you're seeing:</strong> the model's raw target value converted from the dataset's $100,000 units.</div>
                </div>

                <div style="position:absolute;right:-85px;top:-125px;width:245px;height:245px;border-radius:50%;border:1px solid rgba(155,140,255,.08);box-shadow:0 0 0 27px rgba(155,140,255,.014),0 0 0 54px rgba(155,140,255,.010);animation:slowSpin 16s linear infinite;"></div>
            </div>
            <style>
                @keyframes slowSpin {{ from{{transform:rotate(0deg)}} to{{transform:rotate(360deg)}} }}
            </style>
            """,
            height=490,
            scrolling=False,
        )


# ============================================================
# PREDICTION INTERPRETABILITY
# ============================================================

if st.session_state.prediction_inputs is not None:

    st.markdown(
        """
        <div class="section-head">
            <div>
                <div class="section-eyebrow">03 / EXPLAINABILITY</div>
                <h2 class="section-title">How the equation built this prediction</h2>
                <p class="section-desc">A waterfall makes the path from intercept + feature contributions → final model output visible at a glance.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    inp = st.session_state.prediction_inputs.iloc[0]
    coef = pd.Series(model.coef_, index=feature_columns)
    contribution = coef * inp

    waterfall_values = [float(model.intercept_)] + contribution.tolist() + [0.0]
    waterfall_labels = ["Intercept"] + feature_columns + ["Prediction"]
    waterfall_measures = ["absolute"] + ["relative"] * len(feature_columns) + ["total"]

    wf = go.Figure(
        go.Waterfall(
            orientation="v",
            measure=waterfall_measures,
            x=waterfall_labels,
            y=waterfall_values,
            text=[
                f"{v:.3f}" if i < len(waterfall_values) - 1 else f"{st.session_state.prediction:.3f}"
                for i, v in enumerate(waterfall_values)
            ],
            textposition="outside",
            connector=dict(
                line=dict(
                    color="rgba(255,255,255,.14)",
                    width=1,
                    dash="dot",
                )
            ),
            increasing=dict(
                marker=dict(color="#80F0D0"),
            ),
            decreasing=dict(
                marker=dict(color="#FF8B9D"),
            ),
            totals=dict(
                marker=dict(color="#9B8CFF"),
            ),
            hovertemplate="%{x}<br>Step: %{y:.4f}<extra></extra>",
        )
    )

    wf.update_layout(
        height=510,
        margin=dict(l=38, r=24, t=28, b=95),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#aab2c0", family="Manrope"),
        xaxis=dict(
            gridcolor="rgba(255,255,255,0)",
            tickangle=-32,
        ),
        yaxis=dict(
            title="Raw model output",
            gridcolor="rgba(255,255,255,.055)",
            zerolinecolor="rgba(255,255,255,.10)",
        ),
        hoverlabel=dict(
            bgcolor="#10151e",
            bordercolor="rgba(255,255,255,.12)",
            font=dict(color="#fff"),
        ),
        showlegend=False,
    )

    cc1, cc2 = st.columns([1.35, .65], gap="large")

    with cc1:
        st.markdown(
            '<div class="chart-toolbar-note">Scroll = zoom · drag = pan · double-click = reset</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            wf,
            use_container_width=True,
            config={
                "scrollZoom": True,
                "displayModeBar": "hover",
                "displaylogo": False,
                "responsive": True,
                "doubleClick": "reset+autosize",
            },
        )

    with cc2:
        biggest = contribution.abs().sort_values(ascending=False).head(3)

        st.markdown(
            """
            <div class="glass card-pad" style="min-height:100%;">
                <div class="micro">INTERPRETABILITY NOTE</div>
                <div style="font-size:1.1rem;font-weight:800;color:#fff;margin:.55rem 0 .65rem;">
                    Contribution is not causation.
                </div>
                <div style="font-size:.72rem;line-height:1.7;color:#7e8898;">
                    Each step is the product of a fitted coefficient and the
                    supplied feature value. Together with the intercept, those
                    steps sum to the raw prediction.
                </div>
            """,
            unsafe_allow_html=True,
        )

        for name, _ in biggest.items():
            signed = contribution[name]
            sign = "positive" if signed >= 0 else "negative"

            st.markdown(
                f"""
                <div class="insight">
                    <div class="insight-icon">{'+' if signed >= 0 else '−'}</div>
                    <div>
                        <strong style="color:#e9ebf0">{name}</strong>
                        has a {sign} contribution of
                        <strong style="color:#e9ebf0">{signed:.4f}</strong>
                        in raw target units.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        

# ============================================================
# SCENARIO LAB
# ============================================================

st.markdown(
    """
    <div class="section-head">
        <div>
            <div class="section-eyebrow">04 / SCENARIO LAB</div>
            <h2 class="section-title">What happens when one signal changes?</h2>
            <p class="section-desc">Explore the model's response without changing the saved model itself.</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if st.session_state.prediction_inputs is None:

    st.markdown(
        """
        <div class="glass card-pad">
            <div style="color:#8b94a4;font-size:.78rem;">
                Run one prediction above to unlock the interactive scenario lab.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

else:

    scenario_col, delta_col, result_col = st.columns([1.0, 1.2, 1.0], gap="large")

    with scenario_col:
        selected_feature = st.selectbox(
            "Choose a feature",
            options=feature_columns,
            index=feature_columns.index(st.session_state.scenario_feature),
        )
        st.session_state.scenario_feature = selected_feature

    base_value = float(st.session_state.prediction_inputs.iloc[0][selected_feature])

    with delta_col:
        coefficient = float(
            pd.Series(model.coef_, index=feature_columns)[selected_feature]
        )

        span = max(abs(base_value) * 0.45, 1.0)

        delta = st.slider(
            "Change from current value",
            min_value=float(-span),
            max_value=float(span),
            value=float(st.session_state.scenario_delta),
        )

        st.session_state.scenario_delta = delta

    scenario_df = st.session_state.prediction_inputs.copy()
    scenario_df.loc[scenario_df.index[0], selected_feature] = base_value + delta

    scenario_prediction = float(model.predict(scenario_df)[0])
    base_prediction = float(st.session_state.prediction)

    raw_difference = scenario_prediction - base_prediction
    dollar_difference = raw_difference * 100_000

    with result_col:
        direction = "up" if raw_difference >= 0 else "down"
        signed_dollars = f"+${abs(dollar_difference):,.0f}" if dollar_difference >= 0 else f"-${abs(dollar_difference):,.0f}"

        st.markdown(
            f"""
            <div class="glass card-pad" style="height:100%;">
                <div class="micro">SCENARIO RESULT</div>
                <div style="color:#747e90;font-size:.67rem;margin-top:.55rem;">
                    {selected_feature} · {base_value:.4f} → {base_value + delta:.4f}
                </div>
                <div class="value-xl">{scenario_prediction:.4f}</div>
                <div style="color:#8590a0;font-size:.7rem;margin-top:.25rem;">
                    Raw prediction after change
                </div>
                <div class="insight" style="margin-top:1rem;">
                    <div class="insight-icon">{'↑' if direction == 'up' else '↓'}</div>
                    <div>
                        Prediction changed by <strong style="color:#f0f2f5">{signed_dollars}</strong>
                        in displayed dollar terms.
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    scenario_values = np.linspace(base_value - span, base_value + span, 45)

    curve_inputs = pd.concat(
        [st.session_state.prediction_inputs] * len(scenario_values),
        ignore_index=True,
    )
    curve_inputs[selected_feature] = scenario_values

    curve_predictions = model.predict(curve_inputs)

    curve_fig = go.Figure()

    curve_fig.add_trace(
        go.Scatter(
            x=scenario_values,
            y=curve_predictions,
            mode="lines",
            line=dict(color="#9B8CFF", width=3),
            fill="tozeroy",
            fillcolor="rgba(155,140,255,.075)",
            hovertemplate=f"{selected_feature}: %{{x:.3f}}<br>Prediction: %{{y:.4f}}<extra></extra>",
            name="Response curve",
        )
    )

    curve_fig.add_trace(
        go.Scatter(
            x=[base_value],
            y=[base_prediction],
            mode="markers",
            marker=dict(
                size=14,
                color="#80F0D0",
                line=dict(color="#071016", width=3),
            ),
            hovertemplate="Current profile<br>Prediction: %{y:.4f}<extra></extra>",
            name="Current",
        )
    )

    curve_fig.add_trace(
        go.Scatter(
            x=[base_value + delta],
            y=[scenario_prediction],
            mode="markers",
            marker=dict(
                size=13,
                color="#FFC978",
                symbol="diamond",
                line=dict(color="#1a160c", width=3),
            ),
            hovertemplate="Scenario<br>Prediction: %{y:.4f}<extra></extra>",
            name="Scenario",
        )
    )

    curve_fig.update_layout(
        height=450,
        margin=dict(l=28, r=28, t=26, b=38),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#aab2c0", family="Manrope"),
        xaxis=dict(
            title=selected_feature,
            gridcolor="rgba(255,255,255,.055)",
            zerolinecolor="rgba(255,255,255,.08)",
        ),
        yaxis=dict(
            title="Model prediction",
            gridcolor="rgba(255,255,255,.055)",
            zerolinecolor="rgba(255,255,255,.08)",
        ),
        hoverlabel=dict(
            bgcolor="#10151e",
            bordercolor="rgba(255,255,255,.12)",
            font=dict(color="#fff"),
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="right",
            x=1,
            font=dict(color="#7e8998", size=10),
        ),
    )

    st.markdown(
        '<div class="chart-toolbar-note">Scroll = zoom · drag = pan · double-click = reset</div>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        curve_fig,
        use_container_width=True,
        config={
            "scrollZoom": True,
            "displayModeBar": "hover",
            "displaylogo": False,
            "responsive": True,
            "doubleClick": "reset+autosize",
        },
    )

    st.caption(
        f"Because Linear Regression is linear in this feature, the scenario curve is expected "
        f"to be a straight relationship for {selected_feature} while the other inputs stay fixed."
    )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.markdown(
    """
    <div class="section-head">
        <div>
            <div class="section-eyebrow">05 / MODEL BEHAVIOR</div>
            <h2 class="section-title">Where the model lands — and where it misses</h2>
            <p class="section-desc">Two complementary diagnostics: prediction quality and residual behavior.</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if housing is not None:

    perf1, perf2 = st.columns(2, gap="large")

    abs_error = np.abs(y_test - test_pred)

    parity = go.Figure()

    parity.add_trace(
        go.Scattergl(
            x=y_test,
            y=test_pred,
            mode="markers",
            marker=dict(
                size=6,
                opacity=.62,
                color=abs_error,
                colorscale=[
                    [0.00, "#80F0D0"],
                    [0.45, "#9B8CFF"],
                    [1.00, "#FF8B9D"],
                ],
                colorbar=dict(
                    title="Absolute<br>error",
                    tickfont=dict(color="#7f8998"),
                    titlefont=dict(color="#8e98a8"),
                    thickness=11,
                    len=.72,
                    outlinewidth=0,
                ),
                line=dict(width=0),
            ),
            hovertemplate=(
                "<b>Housing sample</b><br>"
                "Actual: %{x:.3f}<br>"
                "Predicted: %{y:.3f}<br>"
                "Absolute error: %{marker.color:.3f}"
                "<extra></extra>"
            ),
        )
    )

    line_min = float(min(y_test.min(), test_pred.min()))
    line_max = float(max(y_test.max(), test_pred.max()))

    parity.add_trace(
        go.Scatter(
            x=[line_min, line_max],
            y=[line_min, line_max],
            mode="lines",
            line=dict(color="#80F0D0", width=2, dash="dot"),
            hoverinfo="skip",
            name="Perfect prediction",
        )
    )

    parity.update_layout(
        height=445,
        margin=dict(l=28, r=30, t=22, b=36),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#aab2c0", family="Manrope"),
        xaxis=dict(
            title="Actual target",
            gridcolor="rgba(255,255,255,.055)",
            zerolinecolor="rgba(255,255,255,.08)",
            fixedrange=False,
        ),
        yaxis=dict(
            title="Predicted target",
            gridcolor="rgba(255,255,255,.055)",
            zerolinecolor="rgba(255,255,255,.08)",
            fixedrange=False,
        ),
        hoverlabel=dict(
            bgcolor="#10151e",
            bordercolor="rgba(255,255,255,.12)",
            font=dict(color="#fff"),
        ),
        showlegend=False,
    )

    residual_scatter = go.Figure()

    residual_scatter.add_trace(
        go.Scattergl(
            x=test_pred,
            y=residuals,
            mode="markers",
            marker=dict(
                size=6,
                opacity=.58,
                color=residuals,
                colorscale=[
                    [0.00, "#FF8B9D"],
                    [0.50, "#9B8CFF"],
                    [1.00, "#80F0D0"],
                ],
                cmid=0,
                line=dict(width=0),
            ),
            hovertemplate=(
                "<b>Residual</b><br>"
                "Predicted: %{x:.3f}<br>"
                "Residual: %{y:.3f}<extra></extra>"
            ),
        )
    )

    residual_scatter.add_hline(
        y=0,
        line_width=2,
        line_dash="dot",
        line_color="#80F0D0",
    )

    if metrics:
        rmse = float(metrics["RMSE"])
        residual_scatter.add_hline(
            y=rmse,
            line_width=1,
            line_dash="dash",
            line_color="rgba(255,201,120,.55)",
            annotation_text="+RMSE",
            annotation_position="top right",
            annotation_font_color="#ffc978",
        )
        residual_scatter.add_hline(
            y=-rmse,
            line_width=1,
            line_dash="dash",
            line_color="rgba(255,201,120,.55)",
            annotation_text="-RMSE",
            annotation_position="bottom right",
            annotation_font_color="#ffc978",
        )

    residual_scatter.update_layout(
        height=445,
        margin=dict(l=28, r=30, t=22, b=36),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#aab2c0", family="Manrope"),
        xaxis=dict(
            title="Predicted target",
            gridcolor="rgba(255,255,255,.055)",
            zerolinecolor="rgba(255,255,255,.08)",
            fixedrange=False,
        ),
        yaxis=dict(
            title="Residual (actual − predicted)",
            gridcolor="rgba(255,255,255,.055)",
            zerolinecolor="rgba(255,255,255,.08)",
            fixedrange=False,
        ),
        hoverlabel=dict(
            bgcolor="#10151e",
            bordercolor="rgba(255,255,255,.12)",
            font=dict(color="#fff"),
        ),
        showlegend=False,
    )

    with perf1:
        st.markdown(
            '<div class="chart-toolbar-note">Scroll = zoom · drag = pan · double-click = reset</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            parity,
            use_container_width=True,
            config={
                "scrollZoom": True,
                "displayModeBar": "hover",
                "displaylogo": False,
                "responsive": True,
                "doubleClick": "reset+autosize",
            },
        )

    with perf2:
        st.markdown(
            '<div class="chart-toolbar-note">Scroll = zoom · drag = pan · double-click = reset</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            residual_scatter,
            use_container_width=True,
            config={
                "scrollZoom": True,
                "displayModeBar": "hover",
                "displaylogo": False,
                "responsive": True,
                "doubleClick": "reset+autosize",
            },
        )

else:
    st.warning(
        "The model is available, but the California Housing dataset could not be loaded "
        "for diagnostics. Prediction itself can still work from the persisted pickle."
    )


# ============================================================
# FEATURE COEFFICIENTS
# ============================================================

st.markdown(
    """
    <div class="section-head">
        <div>
            <div class="section-eyebrow">06 / MODEL DNA</div>
            <h2 class="section-title">The shape of the equation</h2>
            <p class="section-desc">Positive coefficients push the fitted output upward; negative coefficients push it downward, all else equal.</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

coef_series = pd.Series(model.coef_, index=feature_columns)
coef_sorted = coef_series.sort_values()

coef_fig = go.Figure()

for name, value in coef_sorted.items():
    line_color = "#80F0D0" if value >= 0 else "#FF8B9D"

    coef_fig.add_trace(
        go.Scatter(
            x=[0, value],
            y=[name, name],
            mode="lines",
            line=dict(color=line_color, width=5),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    coef_fig.add_trace(
        go.Scatter(
            x=[value],
            y=[name],
            mode="markers",
            marker=dict(
                size=12,
                color=line_color,
                line=dict(color="#080b10", width=3),
            ),
            customdata=[[name, value]],
            hovertemplate="<b>%{customdata[0]}</b><br>Coefficient: %{customdata[1]:.6f}<extra></extra>",
            showlegend=False,
        )
    )

coef_fig.add_vline(
    x=0,
    line_width=1,
    line_color="rgba(255,255,255,.16)",
)

coef_fig.update_layout(
    height=470,
    margin=dict(l=35, r=28, t=22, b=40),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#aab2c0", family="Manrope"),
    xaxis=dict(
        title="Fitted coefficient",
        gridcolor="rgba(255,255,255,.055)",
        zerolinecolor="rgba(255,255,255,.12)",
    ),
    yaxis=dict(
        gridcolor="rgba(255,255,255,0)",
    ),
    hoverlabel=dict(
        bgcolor="#10151e",
        bordercolor="rgba(255,255,255,.12)",
        font=dict(color="#fff"),
    ),
    showlegend=False,
)

st.markdown(
    '<div class="chart-toolbar-note">Scroll = zoom · drag = pan · double-click = reset</div>',
    unsafe_allow_html=True,
)
st.plotly_chart(
    coef_fig,
    use_container_width=True,
    config={
        "scrollZoom": True,
        "displayModeBar": "hover",
        "displaylogo": False,
        "responsive": True,
        "doubleClick": "reset+autosize",
    },
)


# ============================================================
# ML EXPLAINER
# ============================================================

st.markdown(
    """
    <div class="section-head">
        <div>
            <div class="section-eyebrow">07 / HOW IT THINKS</div>
            <h2 class="section-title">From signal to prediction</h2>
            <p class="section-desc">A visual model of the inference path — inputs become a linear equation, which becomes a target value.</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

components.html(
    """
    <div id="ml-pipeline"></div>
    <script type="module">
      import React from "https://esm.sh/react@18.3.1";
      import { createRoot } from "https://esm.sh/react-dom@18.3.1/client";
      import { motion } from "https://esm.sh/framer-motion@11.18.0?deps=react@18.3.1,react-dom@18.3.1";

      const h = React.createElement;

      const cards = [
        {
          n: "01",
          eyebrow: "SIGNALS",
          title: "Property profile",
          text: "Eight numerical signals describe income, age, rooms, occupancy, population, and location.",
          mark: "8"
        },
        {
          n: "02",
          eyebrow: "MODEL",
          title: "Linear equation",
          text: "Each input meets a fitted coefficient, then the intercept completes the equation.",
          mark: "Σ"
        },
        {
          n: "03",
          eyebrow: "OUTPUT",
          title: "MedHouseVal",
          text: "The resulting raw target is expressed in units of $100,000 in the source dataset.",
          mark: "→"
        }
      ];

      function App() {
        return h(
          "div",
          {
            style: {
              display: "grid",
              gridTemplateColumns: "1fr 70px 1fr 70px 1fr",
              alignItems: "center",
              gap: "10px",
              minHeight: "245px",
              padding: "18px",
              borderRadius: "26px",
              border: "1px solid rgba(255,255,255,.09)",
              background:
                "linear-gradient(145deg, rgba(255,255,255,.045), rgba(255,255,255,.012))",
              boxShadow: "0 28px 85px rgba(0,0,0,.26), inset 0 1px rgba(255,255,255,.035)"
            }
          },

          ...cards.flatMap((card, i) => {
            const node = h(
              motion.div,
              {
                key: card.n,
                initial: { opacity: 0, y: 18 },
                animate: { opacity: 1, y: 0 },
                transition: {
                  delay: i * .12,
                  duration: .55,
                  ease: [.22, 1, .36, 1]
                },
                whileHover: {
                  y: -8,
                  scale: 1.02,
                  transition: { duration: .22 }
                },
                style: {
                  minHeight: "165px",
                  display: "flex",
                  flexDirection: "column",
                  justifyContent: "space-between",
                  position: "relative",
                  overflow: "hidden",
                  padding: "16px",
                  borderRadius: "21px",
                  border: "1px solid rgba(255,255,255,.075)",
                  background: "rgba(255,255,255,.018)",
                  boxShadow: "inset 0 1px rgba(255,255,255,.035)"
                }
              },
              h("div", {
                style: {
                  color: "#727d8d",
                  fontFamily: "DM Mono, monospace",
                  fontSize: "9px",
                  letterSpacing: "1.8px"
                }
              }, `${card.n} · ${card.eyebrow}`),

              h("div", {
                style: {
                  width: "42px",
                  height: "42px",
                  display: "grid",
                  placeItems: "center",
                  borderRadius: "14px",
                  color: "#e9ebf1",
                  fontSize: "18px",
                  marginTop: "7px",
                  background: "linear-gradient(145deg, rgba(155,140,255,.13), rgba(128,240,208,.06))",
                  border: "1px solid rgba(255,255,255,.08)"
                }
              }, card.mark),

              h("div", {
                style: {
                  color: "#fff",
                  fontSize: "14px",
                  fontWeight: 800,
                  marginTop: "8px",
                  letterSpacing: "-.02em"
                }
              }, card.title),

              h("div", {
                style: {
                  color: "#7b8595",
                  fontSize: "10px",
                  lineHeight: 1.6,
                  marginTop: "6px"
                }
              }, card.text)
            );

            if (i === 2) return [node];

            return [
              node,
              h(
                motion.div,
                {
                  key: `arrow-${i}`,
                  animate: { x: [0, 5, 0], opacity: [.35, 1, .35] },
                  transition: {
                    duration: 2.1,
                    repeat: Infinity,
                    ease: "easeInOut",
                    delay: i * .2
                  },
                  style: {
                    textAlign: "center",
                    color: i === 0 ? "#9B8CFF" : "#80F0D0",
                    fontSize: "22px",
                    fontFamily: "DM Mono, monospace"
                  }
                },
                "→"
              )
            ];
          })
        );
      }

      createRoot(document.getElementById("ml-pipeline")).render(h(App));
    </script>
    """,
    height=290,
    scrolling=False,
)


# ============================================================
# HISTORY
# ============================================================

if st.session_state.history:

    st.markdown(
        """
        <div class="section-head">
            <div>
                <div class="section-eyebrow">08 / SESSION MEMORY</div>
                <h2 class="section-title">Recent predictions</h2>
                <p class="section-desc">A lightweight local session history for comparing scenarios.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    history_df = pd.DataFrame(st.session_state.history).copy()
    history_df.insert(0, "Run", range(1, len(history_df) + 1))
    history_df["Estimated $"] = history_df["prediction"] * 100_000

    show_cols = [
        "Run",
        "Estimated $",
        "MedInc",
        "HouseAge",
        "AveRooms",
        "AveBedrms",
        "Population",
        "AveOccup",
        "Latitude",
        "Longitude",
    ]

    st.dataframe(
        history_df[show_cols].style.format(
            {
                "Estimated $": "${:,.0f}",
                "MedInc": "{:.2f}",
                "HouseAge": "{:.1f}",
                "AveRooms": "{:.2f}",
                "AveBedrms": "{:.2f}",
                "Population": "{:.0f}",
                "AveOccup": "{:.2f}",
                "Latitude": "{:.4f}",
                "Longitude": "{:.4f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        <div>AURELIA / HOUSING INTELLIGENCE</div>
        <div>PYTHON · SCIKIT-LEARN · STREAMLIT · PLOTLY</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CARD SPOTLIGHT EFFECT
# ============================================================

components.html(
    """
    <script>
    (() => {
      const parent = window.parent;
      const interval = setInterval(() => {
        const doc = parent.document;
        if (!doc) return;

        const cards = doc.querySelectorAll(".glass, [data-testid=\"stForm\"], .result-shell, [data-testid=\"stPlotlyChart\"], .pipeline-node");

        cards.forEach((card) => {
          if (card.dataset.spotlightReady) return;
          card.dataset.spotlightReady = "1";

          card.addEventListener("pointermove", (event) => {
            const rect = card.getBoundingClientRect();
            const x = ((event.clientX - rect.left) / rect.width) * 100;
            const y = ((event.clientY - rect.top) / rect.height) * 100;

            card.style.setProperty("--mx", `${x}%`);
            card.style.setProperty("--my", `${y}%`);
          });
        });
      }, 900);

      setTimeout(() => clearInterval(interval), 30000);
    })();
    </script>
    """,
    height=0,
)
