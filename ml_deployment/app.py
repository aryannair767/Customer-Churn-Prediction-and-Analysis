import os
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import joblib
import shap
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# ── 1. Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Predictor — B2B SaaS",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 2. Inject Custom CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
/* ─── Google Fonts ─────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@300;400;500;600;700&family=Source+Serif+4:wght@400;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

/* ─── CSS Variables (matches portfolio) ────────────────── */
:root {
    --white: #ffffff;
    --off: #f7f7f7;
    --border: #e0e0e0;
    --border-light: #e8e8e8;
    --text: #3d3d3d;
    --muted: #888888;
    --faint: #aaaaaa;
    --accent: #5e81ac;
    --accent-dark: #4a6a90;
    --accent-light: #f0f4f8;
    --amber: #d4a017;
    --red: #bf616a;
    --red-light: #fdf0f0;
    --green: #5b9a6b;
    --green-light: #f0f7f2;
    --sans: 'Source Sans 3', 'Segoe UI', sans-serif;
    --serif: 'Source Serif 4', Georgia, serif;
    --mono: 'IBM Plex Mono', 'SFMono-Regular', Consolas, monospace;
}

/* ─── Global Reset ─────────────────────────────────────── */
html {
    font-size: 18px !important;
}

html, body, [data-testid="stAppViewContainer"] {
    font-family: var(--sans) !important;
    color: var(--text) !important;
    background-color: var(--off) !important;
    font-size: 18px !important;
}

/* Main block container */
.block-container {
    max-width: 100% !important;
    width: 100% !important;
    padding: 2rem 2.5rem 3rem !important;
}

/* Allow main content wrappers to fill 100% width when sidebar is closed */
[data-testid="stAppViewContainer"] > div:nth-child(2) {
    width: 100% !important;
    max-width: 100% !important;
}

[data-testid="stMain"] {
    width: 100% !important;
    max-width: 100% !important;
}

/* ─── Sidebar ──────────────────────────────────────────── */
[data-testid="stSidebar"][aria-expanded="true"] {
    background-color: var(--white) !important;
    border-right: 1px solid var(--border) !important;
    max-width: 33vw !important;
}

[data-testid="stSidebar"][aria-expanded="true"] > div {
    max-width: 33vw !important;
}

[data-testid="stSidebar"][aria-expanded="false"] {
    max-width: 0px !important;
    min-width: 0px !important;
    width: 0px !important;
    padding: 0 !important;
    margin: 0 !important;
    border: none !important;
}

/* Sidebar Collapse & Expand Toggle Button (<< / >>) */
[data-testid="collapsedControl"] button,
[data-testid="stSidebarCollapseButton"] button {
    background: #2563eb !important;
    background-color: #2563eb !important;
    color: #ffffff !important;
    width: 2.6rem !important;
    height: 2.6rem !important;
    min-width: 2.6rem !important;
    min-height: 2.6rem !important;
    border-radius: 8px !important;
    border: none !important;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.45) !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    transition: all 0.2s cubic-bezier(.16,.84,.44,1) !important;
}

[data-testid="collapsedControl"] button:hover,
[data-testid="stSidebarCollapseButton"] button:hover {
    background: #1d4ed8 !important;
    background-color: #1d4ed8 !important;
    transform: scale(1.08) !important;
    box-shadow: 0 6px 16px rgba(37, 99, 235, 0.6) !important;
}

[data-testid="collapsedControl"] button *,
[data-testid="stSidebarCollapseButton"] button * {
    color: #ffffff !important;
    fill: #ffffff !important;
    stroke: #ffffff !important;
}

[data-testid="collapsedControl"] button svg,
[data-testid="stSidebarCollapseButton"] button svg {
    width: 1.3rem !important;
    height: 1.3rem !important;
}

[data-testid="stSidebar"] [data-testid="stMarkdown"] p,
[data-testid="stSidebar"] label {
    font-family: var(--sans) !important;
    color: var(--text) !important;
    font-size: 0.92rem !important;
    font-weight: 500 !important;
}

[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stNumberInput label,
[data-testid="stSidebar"] .stSlider label {
    font-family: var(--sans) !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.02em !important;
    color: var(--text) !important;
    margin-bottom: 0.15rem !important;
}

/* Sidebar section dividers */
.sidebar-section-title {
    font-family: var(--serif) !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
    padding-bottom: 0.35rem !important;
    margin-bottom: 0.75rem !important;
    margin-top: 1.25rem !important;
    letter-spacing: 0.01em !important;
}

/* Sidebar button */
[data-testid="stSidebar"] .stButton > button {
    width: 100% !important;
    background-color: var(--accent) !important;
    color: var(--white) !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: var(--sans) !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 0.7rem 1.25rem !important;
    letter-spacing: 0.02em !important;
    transition: all 0.2s cubic-bezier(.16,.84,.44,1) !important;
    margin-top: 1rem !important;
    cursor: pointer !important;
}

[data-testid="stSidebar"] .stButton > button:hover {
    background-color: var(--accent-dark) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(94,129,172,0.3) !important;
}

[data-testid="stSidebar"] .stButton > button:active {
    transform: translateY(0) !important;
}

/* ─── Hero Banner ──────────────────────────────────────── */
.hero-banner {
    background: var(--white);
    border: 1px solid var(--border);
    border-bottom: 3px solid var(--accent);
    border-radius: 6px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.75rem;
}

.hero-banner h1 {
    font-family: var(--serif) !important;
    font-weight: 700 !important;
    font-size: 1.85rem !important;
    color: var(--text) !important;
    margin: 0 0 0.25rem 0 !important;
    line-height: 1.3 !important;
}

.hero-subtitle {
    font-family: var(--sans) !important;
    font-size: 1.05rem !important;
    color: var(--muted) !important;
    font-weight: 400 !important;
    line-height: 1.6 !important;
    margin: 0 !important;
}

.hero-tag {
    display: inline-block;
    background: var(--accent-light);
    color: var(--accent);
    font-family: var(--mono) !important;
    font-size: 0.75rem !important;
    font-weight: 500;
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    border: 1px solid rgba(94,129,172,0.2);
    margin-top: 0.75rem;
    letter-spacing: 0.03em;
}

/* ─── Stats Row ────────────────────────────────────────── */
.stat-card {
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 1.25rem 1.5rem;
    text-align: center;
    transition: all 0.2s cubic-bezier(.16,.84,.44,1);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100%;
}

.stat-card:hover {
    box-shadow: 0 8px 24px rgba(0,0,0,0.06);
    transform: translateY(-2px);
}

.stat-label {
    font-family: var(--sans) !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    color: var(--muted) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
    margin-bottom: 0.35rem !important;
}

.stat-value {
    font-family: var(--mono) !important;
    font-size: 1.5rem !important;
    font-weight: 600 !important;
    color: var(--text) !important;
    white-space: nowrap !important;
}

.stat-value.accent { color: var(--accent) !important; }

/* Force Streamlit columns holding stat cards to equal height */
[data-testid="stHorizontalBlock"] {
    align-items: stretch !important;
}

[data-testid="stHorizontalBlock"] [data-testid="stColumn"] > div {
    height: 100% !important;
}

[data-testid="stHorizontalBlock"] [data-testid="stColumn"] > div > div {
    height: 100% !important;
}

/* ─── Result Cards ─────────────────────────────────────── */
.result-card {
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 1.75rem 2rem;
    margin-bottom: 1rem;
}

.result-card-title {
    font-family: var(--serif) !important;
    font-weight: 600 !important;
    font-size: 1.15rem !important;
    color: var(--text) !important;
    margin-bottom: 1rem !important;
    padding-bottom: 0.5rem !important;
    border-bottom: 2px solid var(--accent) !important;
}

/* Risk indicator badges */
.risk-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.6rem 1.25rem;
    border-radius: 6px;
    font-family: var(--sans) !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    margin-bottom: 1rem;
}

.risk-badge.high {
    background: var(--red-light);
    color: var(--red);
    border: 1px solid rgba(191,97,106,0.2);
}

.risk-badge.safe {
    background: var(--green-light);
    color: var(--green);
    border: 1px solid rgba(91,154,107,0.2);
}

/* Probability display */
.probability-display {
    background: var(--accent-light);
    border-left: 4px solid var(--accent);
    border-radius: 0 6px 6px 0;
    padding: 1rem 1.5rem;
    margin-top: 0.75rem;
}

.probability-label {
    font-family: var(--sans) !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    color: var(--muted) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
    margin-bottom: 0.2rem !important;
}

.probability-value {
    font-family: var(--mono) !important;
    font-size: 2.25rem !important;
    font-weight: 600 !important;
    line-height: 1.2 !important;
}

.probability-value.high { color: var(--red) !important; }
.probability-value.safe { color: var(--green) !important; }

/* Progress bar */
.prob-bar-bg {
    width: 100%;
    height: 8px;
    background: var(--border-light);
    border-radius: 999px;
    margin-top: 0.75rem;
    overflow: hidden;
}

.prob-bar-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 0.8s cubic-bezier(.16,.84,.44,1);
}

.prob-bar-fill.high { background: var(--red); }
.prob-bar-fill.safe { background: var(--green); }

/* Insight callout (matching portfolio key-insight boxes) */
.insight-callout {
    background: var(--accent-light);
    border-left: 4px solid var(--accent);
    border-radius: 0 6px 6px 0;
    padding: 1rem 1.25rem;
    margin-top: 1rem;
    font-family: var(--sans) !important;
    font-size: 0.9rem !important;
    color: var(--text) !important;
    line-height: 1.6 !important;
}

.insight-callout strong {
    font-weight: 600 !important;
    color: var(--accent-dark) !important;
}

/* ─── Input Summary Table ──────────────────────────────── */
.input-summary-table {
    width: 100%;
    border-collapse: collapse;
    font-family: var(--sans) !important;
    font-size: 0.88rem !important;
    margin-top: 0.5rem;
}

.input-summary-table th {
    text-align: left;
    font-weight: 600;
    color: var(--muted);
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    padding: 0.5rem 0.75rem;
    border-bottom: 2px solid var(--border);
}

.input-summary-table td {
    padding: 0.55rem 0.75rem;
    border-bottom: 1px solid var(--border-light);
    color: var(--text);
}

.input-summary-table tr:last-child td {
    border-bottom: none;
}

.input-summary-table .mono-val {
    font-family: var(--mono) !important;
    font-size: 0.85rem !important;
    font-weight: 500;
    color: var(--accent-dark);
}

/* ─── SHAP Chart Styling ───────────────────────────────── */
.shap-container {
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 1.75rem 2rem;
}

.shap-title {
    font-family: var(--serif) !important;
    font-weight: 600 !important;
    font-size: 1.15rem !important;
    color: var(--text) !important;
    margin-bottom: 0.25rem !important;
    padding-bottom: 0.5rem !important;
    border-bottom: 2px solid var(--accent) !important;
}

.shap-description {
    font-family: var(--sans) !important;
    font-size: 0.88rem !important;
    color: var(--muted) !important;
    margin-bottom: 1.25rem !important;
    line-height: 1.5 !important;
}

/* ─── Welcome State ────────────────────────────────────── */
.welcome-card {
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 3rem 2.5rem;
    text-align: center;
    margin-top: 0.5rem;
}

.welcome-icon {
    font-size: 2.5rem;
    margin-bottom: 1rem;
    opacity: 0.8;
}

.welcome-title {
    font-family: var(--serif) !important;
    font-weight: 600 !important;
    font-size: 1.25rem !important;
    color: var(--text) !important;
    margin-bottom: 0.5rem !important;
}

.welcome-text {
    font-family: var(--sans) !important;
    font-size: 0.92rem !important;
    color: var(--muted) !important;
    line-height: 1.6 !important;
    max-width: 500px;
    margin: 0 auto;
}

.step-list {
    display: flex;
    justify-content: center;
    gap: 2rem;
    margin-top: 1.5rem;
    flex-wrap: wrap;
}

.step-item {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.4rem;
}

.step-number {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    background: var(--accent-light);
    color: var(--accent);
    font-family: var(--mono) !important;
    font-weight: 600;
    font-size: 0.85rem;
    display: flex;
    align-items: center;
    justify-content: center;
    border: 1px solid rgba(94,129,172,0.2);
}

.step-text {
    font-family: var(--sans) !important;
    font-size: 0.82rem !important;
    color: var(--muted) !important;
    font-weight: 500;
}

/* ─── Footer ───────────────────────────────────────────── */
.app-footer {
    margin-top: 2.5rem;
    padding-top: 1rem;
    border-top: 1px solid var(--border);
    text-align: center;
    font-family: var(--sans) !important;
    font-size: 0.78rem !important;
    color: var(--faint) !important;
    letter-spacing: 0.02em;
}

/* ─── Input Styling (clean, minimal — matching selectbox look) ── */
/* Global: kill ALL red/default outlines everywhere */
*:focus-visible,
*:focus {
    outline-color: var(--accent) !important;
}

input:focus,
button:focus,
[role="listbox"]:focus,
[role="combobox"]:focus,
[data-baseweb] *:focus {
    outline-color: var(--accent) !important;
}

/* Number inputs — match selectbox style */
[data-testid="stNumberInput"] input,
[data-testid="stSidebar"] [data-testid="stNumberInput"] input {
    border: 1px solid transparent !important;
    border-radius: 12px !important;
    font-family: var(--sans) !important;
    font-size: 0.92rem !important;
    font-weight: 500 !important;
    color: var(--text) !important;
    background: #f0f2f6 !important;
    padding: 0.55rem 0.85rem !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
    outline: none !important;
}

[data-testid="stNumberInput"] input:focus,
[data-testid="stNumberInput"] input:focus-visible,
[data-testid="stNumberInput"] input:active {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(94,129,172,0.15) !important;
    outline: none !important;
    caret-color: var(--accent) !important;
}

[data-testid="stNumberInput"] input:hover {
    border-color: rgba(0,0,0,0.08) !important;
}

/* Number input +/- step buttons */
[data-testid="stNumberInput"] button,
[data-testid="stSidebar"] [data-testid="stNumberInput"] button {
    border: 1px solid transparent !important;
    color: var(--muted) !important;
    background: #f0f2f6 !important;
    border-radius: 10px !important;
    transition: all 0.15s ease !important;
}

[data-testid="stNumberInput"] button:hover {
    background: #e4e6ea !important;
    color: var(--text) !important;
}

[data-testid="stNumberInput"] button:focus,
[data-testid="stNumberInput"] button:active {
    border-color: transparent !important;
    box-shadow: none !important;
    outline: none !important;
}

/* Selectboxes — same clean style */
[data-testid="stSelectbox"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] > div {
    border: 1px solid transparent !important;
    border-radius: 12px !important;
    font-family: var(--sans) !important;
    font-size: 0.92rem !important;
    font-weight: 500 !important;
    color: var(--text) !important;
    background: #f0f2f6 !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}

[data-testid="stSelectbox"] [data-baseweb="select"] *,
[data-baseweb="select"] span,
[data-baseweb="select"] div[class*="value"],
[data-baseweb="select"] div[class*="single"] {
    color: var(--text) !important;
}

[data-testid="stSelectbox"] [data-baseweb="select"] > div:hover {
    border-color: rgba(0,0,0,0.08) !important;
}

[data-testid="stSelectbox"] [data-baseweb="select"]:focus-within > div {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(94,129,172,0.15) !important;
}

/* Selectbox dropdown list */
[data-baseweb="popover"] ul,
[data-baseweb="menu"] {
    border: 1px solid rgba(0,0,0,0.06) !important;
    border-radius: 12px !important;
    box-shadow: 0 8px 24px rgba(0,0,0,0.08) !important;
}

[data-baseweb="menu"] li:hover,
[data-baseweb="menu"] [role="option"]:hover {
    background: var(--accent-light) !important;
    color: var(--accent-dark) !important;
}

[data-baseweb="menu"] [aria-selected="true"],
[data-baseweb="menu"] [role="option"][aria-selected="true"] {
    background: var(--accent) !important;
    color: var(--white) !important;
}

/* ─── Slider ───────────────────────────────────────────── */
/* Thumb (draggable circle) */
[data-testid="stSlider"] [role="slider"],
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
    background: var(--accent) !important;
    border-color: var(--accent) !important;
    box-shadow: 0 1px 4px rgba(94,129,172,0.3) !important;
}

/* Unfilled track */
[data-testid="stSlider"] [data-baseweb="slider"] div[data-testid="stTickBar"] {
    background: #e4e6ea !important;
    background-color: #e4e6ea !important;
}

/* The inner track bar container */
[data-testid="stSlider"] [data-baseweb="slider"] > div > div:nth-child(3),
[data-testid="stSlider"] [data-baseweb="slider"] > div > div:nth-child(3) > div {
    background-color: #e4e6ea !important;
}

/* The filled part of the track (left of thumb) */
[data-testid="stSlider"] [data-baseweb="slider"] > div > div:nth-child(4),
[data-testid="stSlider"] [data-baseweb="slider"] > div > div:nth-child(4) > div {
    background-color: var(--accent) !important;
}

/* Slider thumb value */
[data-testid="stSlider"] [data-testid="stThumbValue"] {
    color: var(--accent-dark) !important;
    font-family: var(--mono) !important;
    font-weight: 600 !important;
}

/* Text input (if any) */
[data-testid="stTextInput"] input {
    border: 1px solid transparent !important;
    border-radius: 12px !important;
    background: #f0f2f6 !important;
    font-family: var(--sans) !important;
    font-size: 0.92rem !important;
    padding: 0.55rem 0.85rem !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}

[data-testid="stTextInput"] input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(94,129,172,0.15) !important;
    outline: none !important;
}

/* ─── Dark Mode ────────────────────────────────────────── */
[data-theme="dark"] {
    --white: #141414;
    --off: #1a1a1a;
    --border: #2a2a2a;
    --border-light: #333333;
    --text: #e8e8e8;
    --muted: #999999;
    --faint: #555555;
    --accent: #8cb4d8;
    --accent-dark: #7aa4cb;
    --accent-light: #1c2530;
    --red: #e08890;
    --red-light: #291e20;
    --green: #8ec07c;
    --green-light: #1f2721;
}

[data-theme="dark"] html,
[data-theme="dark"] body,
[data-theme="dark"] [data-testid="stAppViewContainer"],
[data-theme="dark"] [data-testid="stApp"] {
    background-color: var(--off) !important;
    color: var(--text) !important;
}

[data-theme="dark"] [data-testid="stSidebar"] {
    background-color: var(--white) !important;
    border-right-color: var(--border) !important;
}

[data-theme="dark"] [data-testid="stSidebar"] label,
[data-theme="dark"] [data-testid="stSidebar"] p {
    color: var(--text) !important;
}

/* Fix white corners peeking through — nuke ALL wrapper backgrounds */
[data-theme="dark"] div[data-baseweb="select"],
[data-theme="dark"] div[data-baseweb="select"] *,
[data-theme="dark"] div[data-baseweb="base-input"],
[data-theme="dark"] [data-testid="stNumberInput"] > div,
[data-theme="dark"] [data-testid="stNumberInput"] > div > div,
[data-theme="dark"] [data-testid="stSelectbox"] > div,
[data-theme="dark"] [data-testid="stSelectbox"] > div > div,
[data-theme="dark"] [data-testid="stSelectbox"] [data-baseweb="select"] div[class],
[data-theme="dark"] [data-baseweb="select"] > div > div {
    background-color: transparent !important;
}

[data-theme="dark"] [data-testid="stNumberInput"] input,
[data-theme="dark"] [data-testid="stSidebar"] [data-testid="stNumberInput"] input {
    background: #222222 !important;
    color: #ffffff !important;
    border: 1px solid transparent !important;
    border-radius: 12px !important;
}

[data-theme="dark"] [data-testid="stNumberInput"] input:focus,
[data-theme="dark"] [data-testid="stSidebar"] [data-testid="stNumberInput"] input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(140, 180, 216, 0.2) !important;
}

[data-theme="dark"] [data-testid="stNumberInput"] button,
[data-theme="dark"] [data-testid="stSidebar"] [data-testid="stNumberInput"] button {
    background: #222222 !important;
    border: 1px solid transparent !important;
    border-radius: 10px !important;
    color: var(--accent) !important;
}

/* Selectboxes — catch ALL of them */
[data-theme="dark"] [data-testid="stSelectbox"] [data-baseweb="select"] > div,
[data-theme="dark"] [data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] > div {
    background: #222222 !important;
    color: #ffffff !important;
    border: 1px solid transparent !important;
    border-radius: 12px !important;
}

/* Force white text on ALL select value containers and dropdown arrows */
[data-theme="dark"] [data-baseweb="select"] *,
[data-theme="dark"] [data-testid="stSelectbox"] * {
    color: #ffffff !important;
}

[data-theme="dark"] [data-baseweb="select"] svg {
    fill: #ffffff !important;
    color: #ffffff !important;
}

[data-theme="dark"] [data-testid="stSelectbox"] [data-baseweb="select"]:focus-within > div,
[data-theme="dark"] [data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"]:focus-within > div {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(140, 180, 216, 0.2) !important;
}

/* Selectbox dropdown menu in dark mode */
[data-theme="dark"] [data-baseweb="popover"],
[data-theme="dark"] [data-baseweb="popover"] *,
[data-theme="dark"] [data-baseweb="menu"],
[data-theme="dark"] [data-baseweb="menu"] *,
[data-theme="dark"] [role="listbox"],
[data-theme="dark"] [role="listbox"] *,
[data-theme="dark"] [role="option"],
[data-theme="dark"] [role="option"] * {
    background: #141414 !important;
    background-color: #141414 !important;
    border-color: #333333 !important;
    color: #ffffff !important;
}

[data-theme="dark"] [data-baseweb="menu"] li:hover,
[data-theme="dark"] [data-baseweb="menu"] li:hover *,
[data-theme="dark"] [role="option"]:hover,
[data-theme="dark"] [role="option"]:hover * {
    background: #1c2530 !important;
    background-color: #1c2530 !important;
    color: #8cb4d8 !important;
}

[data-theme="dark"] [data-baseweb="menu"] [aria-selected="true"],
[data-theme="dark"] [data-baseweb="menu"] [aria-selected="true"] *,
[data-theme="dark"] [role="option"][aria-selected="true"],
[data-theme="dark"] [role="option"][aria-selected="true"] * {
    background: #8cb4d8 !important;
    background-color: #8cb4d8 !important;
    color: #111111 !important;
}

[data-theme="dark"] .hero-banner,
[data-theme="dark"] .stat-card,
[data-theme="dark"] .result-card,
[data-theme="dark"] .welcome-card,
[data-theme="dark"] .shap-container {
    background: var(--white) !important;
    border-color: var(--border) !important;
}

[data-theme="dark"] .hero-banner h1,
[data-theme="dark"] .result-card-title,
[data-theme="dark"] .shap-title,
[data-theme="dark"] .welcome-title,
[data-theme="dark"] .stat-value {
    color: var(--text) !important;
}

[data-theme="dark"] .hero-subtitle,
[data-theme="dark"] .shap-description,
[data-theme="dark"] .welcome-text,
[data-theme="dark"] .stat-label {
    color: var(--muted) !important;
}

[data-theme="dark"] .probability-display,
[data-theme="dark"] .insight-callout {
    background: var(--accent-light) !important;
    border-color: var(--accent) !important;
}

[data-theme="dark"] .input-summary-table td {
    color: var(--text) !important;
    border-bottom-color: var(--border) !important;
}

[data-theme="dark"] .input-summary-table th {
    border-bottom-color: var(--border) !important;
}

[data-theme="dark"] .input-summary-table .mono-val {
    color: var(--accent) !important;
}

[data-theme="dark"] .prob-bar-bg {
    background: var(--border) !important;
}

[data-theme="dark"] .sidebar-section-title {
    color: var(--accent) !important;
    border-bottom-color: var(--accent) !important;
}

[data-theme="dark"] .app-footer {
    border-top-color: var(--border) !important;
    color: var(--faint) !important;
}

[data-theme="dark"] [data-testid="stSidebar"] .stButton > button,
[data-theme="dark"] [data-testid="stSidebar"] .stButton > button * ,
[data-theme="dark"] [data-testid="stSidebar"] .stButton > button p {
    background-color: #6a9fd4 !important;
    color: #ffffff !important;
}

[data-theme="dark"] [data-testid="stSidebar"] .stButton > button:hover {
    background-color: #7db0e0 !important;
    box-shadow: 0 4px 16px rgba(140, 180, 216, 0.35) !important;
}

[data-theme="dark"] .risk-badge.high {
    background: var(--red-light) !important;
    border-color: rgba(191,97,106,0.3) !important;
}

[data-theme="dark"] .risk-badge.safe {
    background: var(--green-light) !important;
    border-color: rgba(163,190,140,0.3) !important;
}

[data-theme="dark"] .step-number {
    background: var(--accent-light) !important;
    border-color: rgba(129,161,193,0.25) !important;
}

[data-theme="dark"] .hero-tag {
    background: var(--accent-light) !important;
    border-color: rgba(129,161,193,0.25) !important;
    color: var(--accent) !important;
}

/* ─── Dark Mode Toggle Button ─────────────────────────── */
.theme-toggle {
    position: absolute;
    top: 0.85rem;
    right: 1rem;
    width: 32px;
    height: 32px;
    border-radius: 50%;
    border: 1px solid var(--border);
    background: var(--off);
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.85rem;
    transition: all 0.25s cubic-bezier(.16,.84,.44,1);
    color: var(--muted);
    z-index: 10;
    padding: 0;
    line-height: 1;
}

.theme-toggle:hover {
    border-color: var(--accent);
    color: var(--accent);
    background: var(--accent-light);
    transform: scale(1.08);
}

/* ─── Streamlit Element Overrides ──────────────────────── */
/* Hide default header decoration */
header[data-testid="stHeader"] {
    background: transparent !important;
}

/* Clean up metric styling */
[data-testid="stMetric"] {
    background: transparent !important;
}

/* Divider */
hr {
    border: none !important;
    border-top: 1px solid var(--border) !important;
    margin: 1.5rem 0 !important;
}

/* Remove Streamlit branding safely without breaking layout */
#MainMenu, footer, .viewerBadge_container__r5tak {
    display: none !important;
}

/* Force sidebar toggle buttons to always be visible, ignoring hover state */
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarHeader"] [data-testid="stSidebarCollapseButton"] {
    opacity: 1 !important;
    visibility: visible !important;
    transition: none !important;
    z-index: 999999 !important;
}

/* Fix any stray default styles */
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    font-family: var(--serif) !important;
    color: var(--text) !important;
}
</style>
""", unsafe_allow_html=True)


# ── 3. Load Model Artifacts ────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'churn_rf_model.pkl')
    columns_path = os.path.join(current_dir, 'model_columns.pkl')
    model = joblib.load(model_path)
    columns = joblib.load(columns_path)
    return model, columns

model, model_columns = load_artifacts()


# ── 4. Hero Banner ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <h1>B2B SaaS Churn Early-Warning System</h1>
    <p class="hero-subtitle">
        Machine learning model that predicts customer churn risk and identifies
        the key drivers behind each prediction using SHAP explainability.
    </p>
    <span class="hero-tag">Random Forest · SHAP · Real-Time Scoring</span>
</div>
""", unsafe_allow_html=True)

# ── 4b. Dark Mode Toggle (via components.html for JS support) ───────────────
components.html("""
<script>
    const parentDoc = window.parent.document;
    
    if (!parentDoc.getElementById('dm-toggle-style')) {
        const style = parentDoc.createElement('style');
        style.id = 'dm-toggle-style';
        style.innerHTML = `
            /* ── Portfolio-style theme toggle ── */
            .dm-toggle {
                position: fixed;
                top: 16px;
                right: 20px;
                width: 36px;
                height: 36px;
                border-radius: 50%;
                border: none;
                background: #ffffff;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                z-index: 999999;
                transition: all 0.35s cubic-bezier(.16,.84,.44,1);
                padding: 0;
                outline: none;
                box-shadow: 0 2px 10px rgba(0,0,0,0.08), 0 0 0 1px rgba(0,0,0,0.04);
            }
            .dm-toggle:hover {
                transform: scale(1.1);
                box-shadow: 0 4px 16px rgba(0,0,0,0.12), 0 0 0 1px rgba(0,0,0,0.06);
            }
            .dm-toggle .icon-sun,
            .dm-toggle .icon-moon {
                position: absolute;
                transition: opacity 0.35s ease, transform 0.45s cubic-bezier(.16,.84,.44,1);
            }
            /* Light mode: show moon, hide sun */
            .dm-toggle .icon-moon {
                opacity: 1;
                transform: scale(1) rotate(0deg);
            }
            .dm-toggle .icon-sun {
                opacity: 0;
                transform: scale(0.5) rotate(-90deg);
            }
            /* Dark mode: dark bg, show sun, hide moon */
            [data-theme="dark"] .dm-toggle {
                background: #1a1a1a;
                box-shadow: 0 2px 10px rgba(0,0,0,0.3), 0 0 0 1px rgba(255,255,255,0.06);
            }
            [data-theme="dark"] .dm-toggle:hover {
                box-shadow: 0 4px 16px rgba(0,0,0,0.4), 0 0 0 1px rgba(255,255,255,0.1);
            }
            [data-theme="dark"] .dm-toggle .icon-sun {
                opacity: 1;
                transform: scale(1) rotate(0deg);
            }
            [data-theme="dark"] .dm-toggle .icon-moon {
                opacity: 0;
                transform: scale(0.5) rotate(90deg);
            }
            /* Sidebar Toggle Button (<< / >>) Global Override */
            [data-testid="collapsedControl"] button,
            [data-testid="stSidebarCollapseButton"] button,
            [data-theme="dark"] [data-testid="collapsedControl"] button,
            [data-theme="dark"] [data-testid="stSidebarCollapseButton"] button {
                background: #2563eb !important;
                background-color: #2563eb !important;
                color: #ffffff !important;
                width: 2.6rem !important;
                height: 2.6rem !important;
                min-width: 2.6rem !important;
                min-height: 2.6rem !important;
                border-radius: 8px !important;
                border: none !important;
                box-shadow: 0 4px 12px rgba(37, 99, 235, 0.45) !important;
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
                transition: all 0.2s cubic-bezier(.16,.84,.44,1) !important;
            }
            [data-testid="collapsedControl"] button:hover,
            [data-testid="stSidebarCollapseButton"] button:hover,
            [data-theme="dark"] [data-testid="collapsedControl"] button:hover,
            [data-theme="dark"] [data-testid="stSidebarCollapseButton"] button:hover {
                background: #1d4ed8 !important;
                background-color: #1d4ed8 !important;
                transform: scale(1.08) !important;
                box-shadow: 0 6px 16px rgba(37, 99, 235, 0.6) !important;
            }
            [data-testid="collapsedControl"] button *,
            button[data-testid="stSidebarCollapseButton"] *,
            [data-theme="dark"] [data-testid="collapsedControl"] button *,
            [data-theme="dark"] button[data-testid="stSidebarCollapseButton"] * {
                color: #ffffff !important;
                fill: #ffffff !important;
                stroke: #ffffff !important;
            }
            [data-testid="collapsedControl"] button svg,
            button[data-testid="stSidebarCollapseButton"] svg,
            [data-theme="dark"] [data-testid="collapsedControl"] button svg,
            [data-theme="dark"] button[data-testid="stSidebarCollapseButton"] svg {
                width: 1.3rem !important;
                height: 1.3rem !important;
            }
            [data-theme="dark"] [data-baseweb="popover"],
            [data-theme="dark"] [data-baseweb="popover"] *,
            [data-theme="dark"] [data-baseweb="menu"],
            [data-theme="dark"] [data-baseweb="menu"] *,
            [data-theme="dark"] [role="listbox"],
            [data-theme="dark"] [role="listbox"] *,
            [data-theme="dark"] [role="option"],
            [data-theme="dark"] [role="option"] * {
                background: #141414 !important;
                background-color: #141414 !important;
                border-color: #333333 !important;
                color: #ffffff !important;
            }
            [data-theme="dark"] [data-baseweb="menu"] li:hover,
            [data-theme="dark"] [data-baseweb="menu"] li:hover *,
            [data-theme="dark"] [role="option"]:hover,
            [data-theme="dark"] [role="option"]:hover * {
                background: #1c2530 !important;
                background-color: #1c2530 !important;
                color: #8cb4d8 !important;
            }
            [data-theme="dark"] [data-baseweb="menu"] [aria-selected="true"],
            [data-theme="dark"] [data-baseweb="menu"] [aria-selected="true"] *,
            [data-theme="dark"] [role="option"][aria-selected="true"],
            [data-theme="dark"] [role="option"][aria-selected="true"] * {
                background: #8cb4d8 !important;
                background-color: #8cb4d8 !important;
                color: #111111 !important;
            }
        `;
        parentDoc.head.appendChild(style);
    }

    let btn = parentDoc.getElementById('dmBtn');
    if (!btn) {
        btn = parentDoc.createElement('button');
        btn.id = 'dmBtn';
        btn.className = 'dm-toggle';
        btn.title = 'Toggle dark mode';
        btn.innerHTML = `
            <span class="icon-sun">
                <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#e8b730" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <circle cx="12" cy="12" r="5" fill="#e8b730"/>
                    <line x1="12" y1="1" x2="12" y2="3"/>
                    <line x1="12" y1="21" x2="12" y2="23"/>
                    <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/>
                    <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/>
                    <line x1="1" y1="12" x2="3" y2="12"/>
                    <line x1="21" y1="12" x2="23" y2="12"/>
                    <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/>
                    <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>
                </svg>
            </span>
            <span class="icon-moon">
                <svg xmlns="http://www.w3.org/2000/svg" width="17" height="17" viewBox="0 0 24 24" fill="#7b8fad" stroke="none">
                    <path d="M12 3a9 9 0 1 0 9 9c0-.46-.04-.92-.1-1.36a5.39 5.39 0 0 1-4.4 2.26 5.4 5.4 0 0 1-3.83-1.59A5.4 5.4 0 0 1 11.1 7.5c0-1.7.78-3.22 2-4.22A9.08 9.08 0 0 0 12 3z"/>
                </svg>
            </span>
        `;
        parentDoc.body.appendChild(btn);

        const html = parentDoc.documentElement;
        
        const saved = localStorage.getItem('churn-app-theme');
        if (saved === 'dark') {
            html.setAttribute('data-theme', 'dark');
        }

        btn.addEventListener('click', function() {
            const current = html.getAttribute('data-theme');
            const next = current === 'dark' ? 'light' : 'dark';
            html.setAttribute('data-theme', next);
            localStorage.setItem('churn-app-theme', next);
        });
    }

    function fixSliderColors() {
        const html = parentDoc.documentElement;
        const isDark = html.getAttribute('data-theme') === 'dark';
        
        if (!isDark) {
            const sliderDivs = parentDoc.querySelectorAll('[data-testid="stSlider"] [data-baseweb="slider"] div[style]');
            sliderDivs.forEach(function(d) {
                if (!d.getAttribute('role')) {
                    d.style.removeProperty('background-color');
                    d.style.removeProperty('background');
                }
            });
            const thumbs = parentDoc.querySelectorAll('[data-testid="stSlider"] [role="slider"]');
            thumbs.forEach(function(t) {
                t.style.removeProperty('background-color');
                t.style.removeProperty('border-color');
            });
            return;
        }

        const accentColor = '#8cb4d8';
        const sliders = parentDoc.querySelectorAll('[data-testid="stSlider"] [data-baseweb="slider"]');
        sliders.forEach(function(slider) {
            const allDivs = slider.querySelectorAll('div[style]');
            allDivs.forEach(function(d) {
                const style = d.getAttribute('style') || '';
                if (style.includes('background') && !d.getAttribute('role')) {
                    const rect = d.getBoundingClientRect();
                    if (rect.height < 12 && rect.height > 0) {
                        d.style.setProperty('background-color', accentColor, 'important');
                        d.style.setProperty('background', accentColor, 'important');
                    }
                }
            });
            const thumbs = slider.querySelectorAll('[role="slider"]');
            thumbs.forEach(function(t) {
                t.style.setProperty('background-color', accentColor, 'important');
                t.style.setProperty('border-color', accentColor, 'important');
            });
        });
    }

    function fixDarkInputs() {
        const html = parentDoc.documentElement;
        const isDark = html.getAttribute('data-theme') === 'dark';
        
        if (!isDark) {
            const elementsToClean = parentDoc.querySelectorAll(
                '[data-baseweb="select"] *, [data-testid="stSelectbox"] *, ' +
                '[data-baseweb="popover"], [data-baseweb="popover"] *, ' +
                '[data-baseweb="menu"], [data-baseweb="menu"] *, ' +
                '[role="listbox"], [role="listbox"] *, ' +
                '[data-testid="stNumberInput"] input'
            );
            elementsToClean.forEach(function(el) {
                el.style.removeProperty('background-color');
                el.style.removeProperty('background');
                el.style.removeProperty('color');
                el.style.removeProperty('border-color');
            });
            return;
        }

        // Fix all selectbox value text & inner elements in dark mode
        const selectValues = parentDoc.querySelectorAll('[data-baseweb="select"] *, [data-testid="stSelectbox"] *');
        selectValues.forEach(function(el) {
            el.style.setProperty('color', '#ffffff', 'important');
        });

        // Fix popover/dropdown backgrounds recursively in dark mode
        const popoverEls = parentDoc.querySelectorAll('[data-baseweb="popover"], [data-baseweb="popover"] *, [data-baseweb="menu"], [data-baseweb="menu"] *, [role="listbox"], [role="listbox"] *');
        popoverEls.forEach(function(el) {
            if (!el.matches(':hover') && el.getAttribute('aria-selected') !== 'true') {
                el.style.setProperty('background-color', '#141414', 'important');
                el.style.setProperty('background', '#141414', 'important');
                el.style.setProperty('color', '#ffffff', 'important');
                el.style.setProperty('border-color', '#333333', 'important');
            }
        });

        // Fix number input text in dark mode
        const numInputs = parentDoc.querySelectorAll('[data-testid="stNumberInput"] input');
        numInputs.forEach(function(el) {
            el.style.setProperty('color', '#ffffff', 'important');
            el.style.setProperty('background', '#111111', 'important');
        });
    }
    
    function fixSidebarToggleButton() {
        // Streamlit sidebar buttons use Material Icon glyphs as text content.
        var allButtons = parentDoc.querySelectorAll('button');
        allButtons.forEach(function(b) {
            var text = b.textContent.trim();
            if (text === 'keyboard_double_arrow_left' || text === 'keyboard_double_arrow_right') {
                b.style.setProperty('background', '#2563eb', 'important');
                b.style.setProperty('background-color', '#2563eb', 'important');
                b.style.setProperty('color', '#ffffff', 'important');
                b.style.setProperty('width', '2.6rem', 'important');
                b.style.setProperty('height', '2.6rem', 'important');
                b.style.setProperty('min-width', '2.6rem', 'important');
                b.style.setProperty('min-height', '2.6rem', 'important');
                b.style.setProperty('border-radius', '8px', 'important');
                b.style.setProperty('border', 'none', 'important');
                b.style.setProperty('box-shadow', '0 4px 12px rgba(37, 99, 235, 0.45)', 'important');
                b.style.setProperty('display', 'flex', 'important');
                b.style.setProperty('align-items', 'center', 'important');
                b.style.setProperty('justify-content', 'center', 'important');
                var children = b.querySelectorAll('*');
                children.forEach(function(c) {
                    c.style.setProperty('color', '#ffffff', 'important');
                    c.style.setProperty('fill', '#ffffff', 'important');
                    c.style.setProperty('stroke', '#ffffff', 'important');
                });
            }
        });
    }

    fixSidebarToggleButton();
    fixSliderColors();
    fixDarkInputs();
    const uiObserver = new MutationObserver(function() { fixSidebarToggleButton(); fixSliderColors(); fixDarkInputs(); });
    uiObserver.observe(parentDoc.body, { childList: true, subtree: true, attributes: true, attributeFilter: ['style', 'data-theme'] });
</script>
""", height=0)


# ── 5. Model Stats Row ────────────────────────────────────────────────────────
n_features = len(model_columns)
n_estimators = model.n_estimators
model_type = "Random Forest"

col_s1, col_s2, col_s3 = st.columns(3)

with col_s1:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-label">Model Type</div>
        <div class="stat-value accent">{model_type}</div>
    </div>
    """, unsafe_allow_html=True)

with col_s2:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-label">Estimators</div>
        <div class="stat-value">{n_estimators}</div>
    </div>
    """, unsafe_allow_html=True)

with col_s3:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-label">Features</div>
        <div class="stat-value">{n_features}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height: 1.25rem'></div>", unsafe_allow_html=True)


# ── 6. Sidebar Inputs ─────────────────────────────────────────────────────────
st.sidebar.markdown("""
<div style="padding: 1rem 0 0.5rem 0;">
    <div style="font-family: 'Source Serif 4', Georgia, serif; font-weight: 700; font-size: 1.35rem; color: var(--text); margin-bottom: 0.15rem;">
        Churn Predictor
    </div>
    <div style="font-family: 'Source Sans 3', sans-serif; font-size: 0.82rem; color: var(--muted); line-height: 1.5;">
        Enter customer attributes to score churn risk in real time.
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown('<div class="sidebar-section-title">Revenue & Engagement</div>', unsafe_allow_html=True)

mrr = st.sidebar.number_input("Monthly Recurring Revenue ($)", min_value=0.0, value=1500.0, step=50.0)
avg_satisfaction_score = st.sidebar.slider("Avg Satisfaction Score", min_value=1.0, max_value=5.0, value=3.5, step=0.1)
total_active_users = st.sidebar.number_input("Total Active Users", min_value=1, value=12)
champion_reliance = st.sidebar.slider("Champion Reliance (%)", min_value=0.0, max_value=100.0, value=25.0, step=1.0)

st.sidebar.markdown('<div class="sidebar-section-title">Support History</div>', unsafe_allow_html=True)

total_tickets = st.sidebar.number_input("Total Support Tickets", min_value=0, value=5)
serious_tickets = st.sidebar.number_input("Serious Tickets", min_value=0, value=1)

st.sidebar.markdown('<div class="sidebar-section-title">Account Details</div>', unsafe_allow_html=True)

plan_tier = st.sidebar.selectbox("Plan Tier", ["starter", "professional", "enterprise"])
billing_cycle = st.sidebar.selectbox("Billing Cycle", ["monthly", "annual"])
industry = st.sidebar.selectbox("Industry", [
    "Technology", "Retail", "Manufacturing", "Finance", "Healthcare",
    "Education", "Real Estate", "Non-Profit", "Logistics", "Media",
    "Government", "Energy", "Hospitality", "Legal"
])
channel = st.sidebar.selectbox("Acquisition Channel", ["organic", "paid", "partner", "referral"])

predict_clicked = st.sidebar.button("Run Prediction", use_container_width=True)


# ── 7. Main Content Area ──────────────────────────────────────────────────────
if predict_clicked:

    # Build feature vector
    input_df = pd.DataFrame(0, index=[0], columns=model_columns)
    input_df['mrr'] = mrr
    input_df['avg_satisfaction_score'] = avg_satisfaction_score
    input_df['total_tickets'] = total_tickets
    input_df['serious_tickets'] = serious_tickets
    input_df['total_active_users'] = total_active_users
    input_df['champion_reliance_percentage'] = champion_reliance

    for col_prefix, val in [
        ('plan_tier', plan_tier),
        ('billing_cycle', billing_cycle),
        ('industry', industry),
        ('acquisition_channel', channel),
    ]:
        col_name = f'{col_prefix}_{val}'
        if col_name in input_df.columns:
            input_df[col_name] = 1

    # Predict
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1] * 100
    risk_class = "high" if prediction == 1 else "safe"

    # ── Results Row ──
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        # Prediction result card
        risk_icon = (
            """<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right:0.25rem; vertical-align:text-bottom;"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>""" 
            if risk_class == "high" 
            else """<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right:0.25rem; vertical-align:text-bottom;"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>"""
        )
        risk_label = "HIGH RISK — Likely to Churn" if risk_class == "high" else "RETAINED — Low Churn Risk"

        st.markdown(f"""
        <div class="result-card">
            <div class="result-card-title">Prediction Result</div>
            <div class="risk-badge {risk_class}">
                <span>{risk_icon}</span>
                <span>{risk_label}</span>
            </div>
            <div class="probability-display">
                <div class="probability-label">Churn Probability</div>
                <div class="probability-value {risk_class}">{probability:.1f}%</div>
                <div class="prob-bar-bg">
                    <div class="prob-bar-fill {risk_class}" style="width: {probability:.1f}%"></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Input summary card
        st.markdown(f"""
        <div class="result-card">
            <div class="result-card-title">Input Summary</div>
            <table class="input-summary-table">
                <thead>
                    <tr><th>Feature</th><th>Value</th></tr>
                </thead>
                <tbody>
                    <tr><td>Monthly Recurring Revenue</td><td class="mono-val">${mrr:,.0f}</td></tr>
                    <tr><td>Satisfaction Score</td><td class="mono-val">{avg_satisfaction_score:.1f} / 5.0</td></tr>
                    <tr><td>Active Users</td><td class="mono-val">{total_active_users}</td></tr>
                    <tr><td>Champion Reliance</td><td class="mono-val">{champion_reliance:.0f}%</td></tr>
                    <tr><td>Total Tickets</td><td class="mono-val">{total_tickets}</td></tr>
                    <tr><td>Serious Tickets</td><td class="mono-val">{serious_tickets}</td></tr>
                    <tr><td>Plan Tier</td><td class="mono-val">{plan_tier.title()}</td></tr>
                    <tr><td>Billing Cycle</td><td class="mono-val">{billing_cycle.title()}</td></tr>
                    <tr><td>Industry</td><td class="mono-val">{industry}</td></tr>
                    <tr><td>Acquisition Channel</td><td class="mono-val">{channel.title()}</td></tr>
                </tbody>
            </table>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        # SHAP explainability card
        st.markdown("""
        <div class="shap-container">
            <div class="shap-title">Risk Driver Analysis</div>
            <div class="shap-description">
                SHAP values show each feature's contribution toward the churn prediction.
                Red bars push toward churn, blue bars push toward retention.
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.spinner("Computing SHAP values..."):
            explainer = shap.TreeExplainer(model)
            shap_obj = explainer(input_df)
            shap_values = shap_obj[:, :, 1]
            shap_values.feature_names = model_columns

            # Build SHAP chart with portfolio-matching style
            matplotlib.rcParams.update({
                'font.family': 'sans-serif',
                'font.sans-serif': ['Source Sans 3', 'Segoe UI', 'Arial'],
                'axes.facecolor': '#ffffff',
                'figure.facecolor': '#ffffff',
                'text.color': '#3d3d3d',
                'axes.labelcolor': '#3d3d3d',
                'xtick.color': '#888888',
                'ytick.color': '#3d3d3d',
                'axes.edgecolor': '#e0e0e0',
                'grid.color': '#e8e8e8',
            })

            fig, ax = plt.subplots(figsize=(9, 4.5))
            shap.plots.waterfall(shap_values[0], max_display=8, show=False)

            # Style the chart axes
            for spine in ax.spines.values():
                spine.set_color('#e0e0e0')
                spine.set_linewidth(0.75)

            ax.tick_params(axis='both', labelsize=9)
            plt.tight_layout(pad=1.5)
            st.pyplot(fig, use_container_width=True)

        # Insight callout
        if risk_class == "high":
            insight_text = (
                "<strong>Key Insight:</strong> The model flags this customer as at-risk. "
                "Review the top contributing factors above — addressing the strongest "
                "churn drivers (red bars) can significantly improve retention odds."
            )
        else:
            insight_text = (
                "<strong>Key Insight:</strong> This customer shows healthy retention signals. "
                "The features pushing toward retention (blue bars) outweigh churn risk factors. "
                "Continue monitoring satisfaction and engagement metrics."
            )

        st.markdown(f'<div class="insight-callout">{insight_text}</div>', unsafe_allow_html=True)

else:
    # ── Welcome / Empty State ──
    st.markdown("""
    <div class="welcome-card">
        <div class="welcome-icon">
            <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><line x1="3" y1="9" x2="21" y2="9"></line><line x1="9" y1="21" x2="9" y2="9"></line></svg>
        </div>
        <div class="welcome-title">Ready to Analyze</div>
        <div class="welcome-text">
            Configure customer attributes in the sidebar and run a prediction
            to see churn risk scores with full SHAP explainability.
        </div>
        <div class="step-list">
            <div class="step-item">
                <div class="step-number">1</div>
                <div class="step-text">Enter data</div>
            </div>
            <div class="step-item">
                <div class="step-number">2</div>
                <div class="step-text">Run prediction</div>
            </div>
            <div class="step-item">
                <div class="step-number">3</div>
                <div class="step-text">Analyze drivers</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── 8. Footer ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
    Built with Streamlit · Random Forest Classifier · SHAP Explainability
</div>
""", unsafe_allow_html=True)
