from __future__ import annotations

import threading
import time
from io import BytesIO

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

from model_utils import (
    CHECKPOINT_DIR,
    ModelBundle,
    debug_predict_image,
    detect_and_crop_face,
    draw_face_box,
    get_device,
    load_model_bundle,
    preprocessing_tensor_stats,
)

try:
    import av
    from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

    WEBRTC_AVAILABLE = True
    WEBRTC_IMPORT_ERROR = None
except Exception as exc:
    WEBRTC_AVAILABLE = False
    WEBRTC_IMPORT_ERROR = exc
    av = None
    VideoProcessorBase = object
    WebRtcMode = None
    webrtc_streamer = None


st.set_page_config(
    page_title="Facial Emotion Recognition",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)


APP_CACHE_VERSION = "crop-square-weighted-ensemble-v2"


@st.cache_resource(show_spinner=False)
def cached_model_bundle(mode: str, cache_version: str = APP_CACHE_VERSION) -> ModelBundle:
    return load_model_bundle(mode=mode, device=get_device())


def apply_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #04070D;
            --bg-2: #08111F;
            --panel: #0D1726;
            --card: rgba(15, 23, 42, 0.72);
            --text: #F8FAFC;
            --text-2: #CBD5E1;
            --muted: #94A3B8;
            --border: rgba(148, 163, 184, 0.18);
            --active: rgba(34, 211, 238, 0.45);
            --cyan: #22D3EE;
            --blue: #3B82F6;
            --violet: #8B5CF6;
            --teal: #14B8A6;
            --page-x: clamp(1.25rem, 3vw, 2.5rem);
            --max-page: 1400px;
        }
        #MainMenu, footer, header { visibility: hidden; }
        .stApp {
            background:
                linear-gradient(rgba(148,163,184,0.028) 1px, transparent 1px),
                linear-gradient(90deg, rgba(148,163,184,0.028) 1px, transparent 1px),
                radial-gradient(circle at 85% 10%, rgba(59,130,246,0.18), transparent 28%),
                radial-gradient(circle at 10% 70%, rgba(34,211,238,0.10), transparent 25%),
                linear-gradient(135deg, #04070D 0%, #08111F 45%, #050816 100%);
            background-size: 42px 42px, 42px 42px, auto, auto, auto;
            color: var(--text);
        }
        .block-container,
        [data-testid="stMainBlockContainer"] {
            max-width: var(--max-page) !important;
            width: 100% !important;
            margin: 0 auto !important;
            padding: 1.5rem var(--page-x) 4rem !important;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(8,17,31,0.98), rgba(4,7,13,0.98));
            border-right: 1px solid var(--border);
        }
        [data-testid="stSidebar"] * { color: var(--text-2); }
        .hero {
            position: relative;
            overflow: hidden;
            margin: 0 0 2.2rem;
            padding: clamp(2rem, 4vw, 3.4rem);
            border: 1px solid var(--border);
            border-radius: 30px;
            background:
                radial-gradient(circle at 78% 10%, rgba(34,211,238,0.12), transparent 30%),
                radial-gradient(circle at 0% 100%, rgba(59,130,246,0.10), transparent 26%),
                linear-gradient(145deg, rgba(15,23,42,0.86), rgba(8,17,31,0.78));
            box-shadow: 0 24px 70px rgba(0,0,0,0.42);
            animation: fadeIn 600ms ease-out;
        }
        .hero-grid {
            position: relative;
            z-index: 1;
            display: grid;
            grid-template-columns: minmax(0, 1.08fr) minmax(380px, 0.82fr);
            gap: clamp(2rem, 5vw, 4.5rem);
            align-items: center;
        }
        .hero::before {
            content: "";
            position: absolute;
            inset: 0;
            background:
                linear-gradient(90deg, transparent, rgba(34,211,238,0.04), transparent),
                repeating-linear-gradient(0deg, rgba(248,250,252,0.035) 0 1px, transparent 1px 9px);
            opacity: 0.6;
            pointer-events: none;
        }
        .hero-copy, .hero-visual { position: relative; z-index: 1; }
        .eyebrow {
            color: var(--cyan);
            font-size: 0.82rem;
            letter-spacing: 0.15em;
            text-transform: uppercase;
            margin-bottom: 1rem;
            font-weight: 700;
        }
        .hero-title {
            max-width: 760px;
            font-size: clamp(3rem, 6vw, 5.8rem);
            line-height: 0.96;
            font-weight: 800;
            letter-spacing: 0;
            margin: 0;
            background: linear-gradient(90deg, #FFFFFF, #DFF8FF 42%, #B8C7FF 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .hero-subtitle {
            max-width: 760px;
            font-size: 1.18rem;
            line-height: 1.65;
            color: var(--text-2);
            margin-top: 1.35rem;
        }
        .supporting-line {
            color: var(--muted);
            font-size: 0.96rem;
            margin-top: 0.7rem;
        }
        .metric-strip {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.75rem;
            margin-top: 1.8rem;
            max-width: 820px;
        }
        .metric-badge {
            padding: 0.8rem 0.9rem;
            border-radius: 16px;
            border: 1px solid rgba(34,211,238,0.22);
            background: rgba(4,7,13,0.38);
        }
        .metric-badge span {
            display: block;
            color: var(--muted);
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.25rem;
        }
        .metric-badge strong {
            color: var(--text);
            font-size: 0.98rem;
        }
        .tech-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            margin-top: 1.3rem;
        }
        .tech-pill {
            padding: 0.46rem 0.72rem;
            border-radius: 999px;
            border: 1px solid rgba(148,163,184,0.20);
            background: rgba(15,23,42,0.52);
            color: var(--text-2);
            font-size: 0.84rem;
        }
        .hero-visual-card, .status-panel {
            position: relative;
            overflow: hidden;
            min-height: 410px;
            padding: 1.35rem;
            border-radius: 26px;
            border: 1px solid rgba(34,211,238,0.28);
            background:
                linear-gradient(145deg, rgba(13,23,38,0.92), rgba(4,7,13,0.84)),
                radial-gradient(circle at 70% 0%, rgba(34,211,238,0.13), transparent 34%);
            box-shadow: 0 0 0 1px rgba(34,211,238,0.08), 0 24px 60px rgba(0,0,0,0.45);
            animation: slideUp 700ms ease-out;
        }
        .status-panel::before {
            content: "";
            position: absolute;
            inset: 0;
            background-image:
                linear-gradient(rgba(34,211,238,0.06) 1px, transparent 1px),
                linear-gradient(90deg, rgba(34,211,238,0.05) 1px, transparent 1px);
            background-size: 24px 24px;
            mask-image: linear-gradient(180deg, rgba(0,0,0,0.85), transparent 86%);
            pointer-events: none;
        }
        .status-panel::after {
            content: "";
            position: absolute;
            left: 0;
            right: 0;
            top: -20%;
            height: 34%;
            background: linear-gradient(180deg, transparent, rgba(34,211,238,0.08), transparent);
            animation: scanLine 5s linear infinite;
            pointer-events: none;
        }
        .preview-top {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            align-items: flex-start;
            margin-bottom: 1.1rem;
        }
        .live-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            white-space: nowrap;
            padding: 0.42rem 0.7rem;
            border-radius: 999px;
            color: #BFF9FF;
            background: rgba(20,184,166,0.10);
            border: 1px solid rgba(20,184,166,0.32);
            font-size: 0.82rem;
        }
        .live-pill span {
            width: 0.5rem;
            height: 0.5rem;
            border-radius: 999px;
            background: var(--teal);
            box-shadow: 0 0 16px rgba(20,184,166,0.75);
            animation: softPulse 1.8s ease-in-out infinite;
        }
        .status-readout {
            position: relative;
            z-index: 1;
            margin-top: 1.1rem;
            padding: 1rem;
            border-radius: 20px;
            background: rgba(4,7,13,0.42);
            border: 1px solid rgba(148,163,184,0.14);
        }
        .waveform {
            display: flex;
            align-items: end;
            gap: 0.35rem;
            height: 96px;
            margin: 0.9rem 0 1rem;
        }
        .waveform span {
            flex: 1;
            min-width: 7px;
            border-radius: 999px;
            background: linear-gradient(180deg, rgba(34,211,238,0.85), rgba(59,130,246,0.22));
            opacity: 0.85;
        }
        .status-list {
            position: relative;
            z-index: 1;
            display: grid;
            gap: 0.68rem;
        }
        .visual-stats {
            display: grid;
            grid-template-columns: 1fr;
            gap: 0.75rem;
        }
        .visual-stat {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.82rem 0.9rem;
            border-radius: 16px;
            background: rgba(15,23,42,0.58);
            border: 1px solid rgba(148,163,184,0.16);
        }
        .visual-stat strong {
            color: var(--text);
            font-size: 0.96rem;
        }
        .visual-stat span {
            color: var(--muted);
            font-size: 0.86rem;
        }
        .section-title {
            font-size: 1.85rem;
            font-weight: 700;
            margin: 3rem 0 0.8rem;
            letter-spacing: 0;
        }
        .section-subtitle {
            color: var(--text-2);
            margin: -0.25rem 0 1.35rem;
            max-width: 880px;
            font-size: 1rem;
            line-height: 1.55;
        }
        .premium-card, .metric-card, .result-card {
            border: 1px solid var(--border);
            background: linear-gradient(145deg, rgba(15,23,42,0.86), rgba(8,17,31,0.78));
            border-radius: 22px;
            padding: 1.15rem;
            box-shadow: 0 18px 50px rgba(0,0,0,0.35);
            transition: transform 180ms ease, border-color 180ms ease, box-shadow 180ms ease;
        }
        .premium-card:hover, .metric-card:hover {
            transform: translateY(-2px);
            border-color: rgba(34,211,238,0.28);
        }
        .card-title {
            color: var(--text);
            font-size: 1.05rem;
            font-weight: 700;
            margin-bottom: 0.55rem;
        }
        .card-text {
            color: var(--text-2);
            font-size: 0.95rem;
            line-height: 1.5;
        }
        .overview-grid {
            display: grid;
            grid-template-columns: minmax(0, 1.05fr) minmax(320px, 0.72fr);
            gap: 1rem;
            align-items: stretch;
        }
        .overview-panel {
            border: 1px solid var(--border);
            border-radius: 24px;
            padding: 1.35rem;
            background: linear-gradient(145deg, rgba(15,23,42,0.86), rgba(8,17,31,0.78));
            box-shadow: 0 18px 50px rgba(0,0,0,0.35);
        }
        .overview-panel p { color: var(--text-2); line-height: 1.58; margin: 0.45rem 0; }
        .timeline {
            display: grid;
            gap: 0.7rem;
            padding-top: 0.2rem;
        }
        .timeline-step {
            display: grid;
            grid-template-columns: 22px 1fr;
            gap: 0.7rem;
            align-items: start;
            color: var(--text-2);
        }
        .timeline-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-top: 0.28rem;
            background: var(--cyan);
            box-shadow: 0 0 0 4px rgba(34,211,238,0.08);
        }
        .demo-shell {
            padding: 1.25rem;
            border: 1px solid rgba(34,211,238,0.26);
            border-radius: 28px;
            background:
                radial-gradient(circle at 12% 0%, rgba(34,211,238,0.10), transparent 30%),
                linear-gradient(145deg, rgba(13,23,38,0.92), rgba(4,7,13,0.84));
            box-shadow: 0 24px 72px rgba(0,0,0,0.44);
            margin: 0.75rem 0 0;
        }
        .demo-intro {
            display: grid;
            grid-template-columns: minmax(0, 1fr) auto;
            gap: 1rem;
            align-items: center;
            padding: 0.35rem 0.2rem 0.85rem;
            color: var(--text-2);
        }
        .demo-kicker {
            color: var(--cyan);
            font-size: 0.78rem;
            letter-spacing: 0.13em;
            text-transform: uppercase;
            margin-bottom: 0.4rem;
        }
        .demo-title {
            color: var(--text);
            font-size: clamp(1.55rem, 2.5vw, 2.2rem);
            font-weight: 760;
            line-height: 1.1;
            margin-bottom: 0.45rem;
        }
        .demo-note {
            max-width: 420px;
            color: var(--muted);
            font-size: 0.92rem;
            text-align: right;
        }
        .dashboard-grid, .metric-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 1rem;
        }
        .dashboard-card {
            min-height: 190px;
            border: 1px solid var(--border);
            background: linear-gradient(145deg, rgba(15,23,42,0.82), rgba(8,17,31,0.74));
            border-radius: 22px;
            padding: 1.2rem;
            box-shadow: 0 18px 50px rgba(0,0,0,0.32);
        }
        .dashboard-card.wide {
            grid-column: span 2;
            min-height: 398px;
        }
        .pipeline-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.65rem;
            margin-top: 1.1rem;
        }
        .pipeline-node {
            flex: 1 1 120px;
            text-align: center;
            padding: 0.8rem 0.75rem;
            border-radius: 14px;
            color: var(--text-2);
            background: rgba(4,7,13,0.34);
            border: 1px solid var(--border);
        }
        .mode-list {
            display: grid;
            gap: 0.65rem;
            margin-top: 1rem;
        }
        .mode-row {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            align-items: center;
            padding: 0.85rem 0.9rem;
            border-radius: 14px;
            background: rgba(4,7,13,0.36);
            border: 1px solid var(--border);
            color: var(--text-2);
        }
        .mode-row strong {
            color: var(--text);
            white-space: nowrap;
        }
        .metric-value {
            font-size: 2rem;
            font-weight: 800;
            color: var(--text);
        }
        .metric-label {
            color: var(--muted);
            margin-top: 0.3rem;
        }
        .metric-card {
            border-color: rgba(34,211,238,0.18);
        }
        .result-label {
            font-size: 2.8rem;
            font-weight: 820;
            color: var(--text);
            margin: 0.25rem 0;
        }
        .result-card {
            border: 1px solid rgba(34,211,238,0.38);
            box-shadow: 0 0 0 1px rgba(34,211,238,0.08), 0 24px 60px rgba(0,0,0,0.45);
        }
        .confidence-pill {
            display: inline-block;
            padding: 0.45rem 0.75rem;
            border-radius: 999px;
            background: rgba(34,211,238,0.08);
            color: #BFF9FF;
            border: 1px solid rgba(34,211,238,0.30);
        }
        .mini-card {
            padding: 0.78rem 0.9rem;
            border-radius: 14px;
            border: 1px solid var(--border);
            background: rgba(15,23,42,0.60);
            margin-bottom: 0.65rem;
        }
        .preview-frame {
            max-width: 560px;
            margin: 0 auto 1rem;
        }
        .crop-frame {
            max-width: 340px;
            margin: 0 auto 1rem;
        }
        .preview-frame img,
        .crop-frame img {
            max-height: 360px !important;
            width: auto !important;
            max-width: 100% !important;
            object-fit: contain !important;
        }
        .camera-helper-label {
            width: min(100%, 820px);
            margin: 1.15rem auto 0.65rem;
            color: var(--text);
            font-weight: 700;
        }
        [data-testid="stCameraInput"] {
            width: min(100%, 820px) !important;
            max-width: 820px !important;
            margin: 0 auto 1.35rem !important;
            border-radius: 22px !important;
            overflow: hidden !important;
        }
        [data-testid="stCameraInput"] video,
        [data-testid="stCameraInput"] img,
        [data-testid="stCameraInput"] canvas {
            width: 100% !important;
            height: auto !important;
            max-height: 360px !important;
            aspect-ratio: 16 / 9 !important;
            object-fit: cover !important;
            object-position: center center !important;
            border-radius: 20px !important;
        }
        [data-testid="stCameraInput"] button {
            position: relative !important;
            z-index: 2 !important;
            margin-top: 0.45rem !important;
        }
        iframe[title*="streamlit_webrtc"],
        iframe[title*="streamlit-webrtc"] {
            max-width: 720px !important;
            max-height: 470px !important;
            display: block !important;
            margin: 0 auto !important;
            border-radius: 20px !important;
        }
        [data-testid="stImage"] img {
            border-radius: 18px;
            border: 1px solid var(--border);
            box-shadow: 0 18px 42px rgba(0,0,0,0.32);
            max-height: 520px;
            object-fit: contain;
        }
        .footer {
            margin-top: 4rem;
            color: var(--muted);
            text-align: left;
            font-size: 0.95rem;
            border-top: 1px solid var(--border);
            padding-top: 1.25rem;
        }
        div[data-testid="stTabs"] {
            padding: 0 0 0.4rem;
            margin: 1rem 0 0;
        }
        div[data-testid="stTabs"] [data-baseweb="tab-list"] {
            gap: 0.35rem;
            background:
                linear-gradient(180deg, rgba(15,23,42,0.82), rgba(8,17,31,0.74));
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 0.42rem;
            box-shadow: none;
        }
        div[data-testid="stTabs"] [data-baseweb="tab"] {
            min-height: 42px;
            padding: 0.68rem 0.95rem;
            color: var(--text-2);
            border-radius: 14px;
            border: 1px solid transparent;
            transition: transform 180ms ease, background 180ms ease, color 180ms ease, border-color 180ms ease, box-shadow 180ms ease;
        }
        div[data-testid="stTabs"] [data-baseweb="tab"]:hover {
            color: var(--text);
            transform: translateY(-1px);
            background: rgba(15,23,42,0.82);
            border-color: var(--border);
        }
        div[data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"] {
            color: var(--text);
            background: linear-gradient(135deg, rgba(34,211,238,0.13), rgba(59,130,246,0.10));
            border-color: rgba(34,211,238,0.42);
            box-shadow: 0 0 0 1px rgba(34,211,238,0.06), 0 10px 28px rgba(0,0,0,0.22);
            animation: borderGlow 2.4s ease-in-out infinite alternate;
        }
        div[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
            height: 2px;
            border-radius: 999px;
            background: linear-gradient(90deg, var(--cyan), var(--teal));
        }
        div[data-baseweb="radio"] {
            background: rgba(15,23,42,0.55);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 0.65rem 0.8rem;
        }
        [data-testid="stFileUploader"] {
            border: 1px dashed rgba(34,211,238,0.24);
            border-radius: 18px;
            padding: 0.4rem;
            background: rgba(4,7,13,0.32);
        }
        [data-testid="stFileUploader"] section {
            background: rgba(15,23,42,0.54);
            border: 0;
            border-radius: 16px;
        }
        .stButton > button {
            border-radius: 999px;
            border: 1px solid rgba(34,211,238,0.36);
            background: rgba(34,211,238,0.08);
            color: var(--text);
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(8px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @keyframes slideUp {
            from { opacity: 0; transform: translateY(14px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @keyframes softPulse {
            0%, 100% { opacity: 0.65; transform: scale(0.96); }
            50% { opacity: 1; transform: scale(1.08); }
        }
        @keyframes scanLine {
            from { transform: translateY(0); }
            to { transform: translateY(420%); }
        }
        @keyframes borderGlow {
            from { border-color: rgba(34,211,238,0.34); }
            to { border-color: rgba(20,184,166,0.46); }
        }
        @media (max-width: 900px) {
            .metric-strip { grid-template-columns: repeat(2, minmax(0, 1fr)); }
            .overview-grid, .dashboard-grid, .metric-grid { grid-template-columns: 1fr; }
            .dashboard-card.wide { grid-column: auto; min-height: auto; }
            .hero-grid { grid-template-columns: 1fr; }
            .hero { padding: 2rem 1.2rem; }
            .demo-intro { display: block; }
            .demo-note { text-align: left; margin-top: 0.8rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero_section() -> None:
    st.markdown(
        """
        <section class="hero">
            <div class="hero-grid">
                <div class="hero-copy">
                    <div class="eyebrow">CSE 676 Deep Learning Demo</div>
                    <h1 class="hero-title">Facial Emotion Recognition</h1>
                    <div class="hero-subtitle">
                        A real-time emotion recognition system trained on FER-2013 using CNNs, transfer learning, CLIP, and ensemble inference.
                    </div>
                    <div class="supporting-line">
                        This app crops the detected face and runs the selected saved checkpoint on the crop. Predictions can vary with lighting, pose, and crop quality.
                    </div>
                    <div class="metric-strip">
                        <div class="metric-badge"><span>Best Accuracy</span><strong>70.23%</strong></div>
                        <div class="metric-badge"><span>Classes</span><strong>7</strong></div>
                        <div class="metric-badge"><span>Dataset</span><strong>FER-2013</strong></div>
                        <div class="metric-badge"><span>Modes</span><strong>Image / Webcam / Live</strong></div>
                    </div>
                    <div class="tech-row">
                        <span class="tech-pill">PyTorch</span>
                        <span class="tech-pill">Streamlit</span>
                        <span class="tech-pill">ConvNeXt</span>
                        <span class="tech-pill">CLIP</span>
                        <span class="tech-pill">FER-2013</span>
                    </div>
                </div>
                <div class="hero-visual status-panel">
                    <div class="preview-top">
                        <div>
                            <div class="card-title">Model Status Panel</div>
                            <div class="card-text">Saved-model inference with face crop input and seven-class probability output.</div>
                        </div>
                        <div class="live-pill"><span></span> Local inference</div>
                    </div>
                    <div class="status-readout">
                        <div class="card-text">Probability trace</div>
                        <div class="waveform">
                            <span style="height: 44%;"></span>
                            <span style="height: 68%;"></span>
                            <span style="height: 52%;"></span>
                            <span style="height: 86%;"></span>
                            <span style="height: 64%;"></span>
                            <span style="height: 76%;"></span>
                            <span style="height: 48%;"></span>
                            <span style="height: 72%;"></span>
                            <span style="height: 58%;"></span>
                            <span style="height: 82%;"></span>
                        </div>
                    </div>
                    <div class="status-list">
                        <div class="visual-stat"><span>Active model</span><strong>CLIP + cleaned-data ensemble</strong></div>
                        <div class="visual-stat"><span>Fast mode</span><strong>ConvNeXt-Tiny</strong></div>
                        <div class="visual-stat"><span>Input</span><strong>Face crop</strong></div>
                        <div class="visual-stat"><span>Output</span><strong>7 emotion probabilities</strong></div>
                    </div>
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def project_story_section() -> None:
    st.markdown('<div class="section-title">Project Overview</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">The project compares simple CNN training, transfer learning, modern CNN backbones, CLIP, and soft-voting ensembles on FER-2013.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="overview-grid">
            <div class="overview-panel">
                <div class="card-title">What this project tested</div>
                <p><strong>Problem.</strong> Facial emotion recognition is difficult when expressions are subtle and images are low resolution.</p>
                <p><strong>Dataset.</strong> FER-2013 contains about 35k 48x48 grayscale face images across Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise.</p>
                <p><strong>Approach.</strong> We compared a baseline CNN, ResNet-18, ConvNeXt-Tiny, CLIP, cleaned-data training, test-time augmentation, and probability averaging.</p>
                <p><strong>Result.</strong> The best saved result was CLIP combined with the cleaned-data ensemble at 70.23% test accuracy.</p>
            </div>
            <div class="overview-panel">
                <div class="card-title">Model progression</div>
                <div class="timeline">
                    <div class="timeline-step"><div class="timeline-dot"></div><div><strong>Baseline CNN</strong><br><span>Small model trained from scratch.</span></div></div>
                    <div class="timeline-step"><div class="timeline-dot"></div><div><strong>ResNet-18</strong><br><span>ImageNet transfer learning and fine-tuning.</span></div></div>
                    <div class="timeline-step"><div class="timeline-dot"></div><div><strong>ConvNeXt-Tiny</strong><br><span>Stronger CNN branch for Fast Mode.</span></div></div>
                    <div class="timeline-step"><div class="timeline-dot"></div><div><strong>CLIP</strong><br><span>Foundation-model branch used in Best Mode.</span></div></div>
                    <div class="timeline-step"><div class="timeline-dot"></div><div><strong>Final ensemble</strong><br><span>Average saved-model probabilities.</span></div></div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def performance_section() -> None:
    st.markdown('<div class="section-title">Performance Snapshot</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">A compact view of the main saved results used to frame the demo. The app still uses checkpoints for real inference and does not hard-code predictions.</div>',
        unsafe_allow_html=True,
    )
    metrics = [
        ("ResNet-18 Fine-Tuning", "66.6%"),
        ("Cleaned-data Ensemble", "69.77%"),
        ("CLIP + Cleaned-data Ensemble", "70.23%"),
    ]
    metric_cards = "\n".join(
        f"""
        <div class="metric-card">
            <div class="metric-value">{value}</div>
            <div class="metric-label">{name}</div>
        </div>
        """
        for name, value in metrics
    )
    st.markdown(f'<div class="metric-grid">{metric_cards}</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-subtitle" style="margin-top:1rem;">The ensemble performed best because it combines CNN-based features with CLIP\'s foundation-model representation.</p>',
        unsafe_allow_html=True,
    )


def model_dashboard_section() -> None:
    st.markdown('<div class="section-title">System Notes</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="section-subtitle">
            The interface keeps the saved-model pipeline visible: crop the face, transform the crop, run the selected model path, then inspect probabilities.
        </div>
        <div class="dashboard-grid">
            <div class="dashboard-card wide">
                <div class="card-title">Inference Pipeline</div>
                <div class="card-text">Image upload, webcam photo, and live camera share the same face-crop-first workflow before prediction.</div>
                <div class="pipeline-row">
                    <div class="pipeline-node">Image</div>
                    <div class="pipeline-node">Face Crop</div>
                    <div class="pipeline-node">Transform</div>
                    <div class="pipeline-node">Model</div>
                    <div class="pipeline-node">Probabilities</div>
                </div>
                <div class="mode-list">
                    <div class="mode-row"><span>Face handling</span><strong>Largest detected face</strong></div>
                    <div class="mode-row"><span>Fallback</span><strong>Full image</strong></div>
                    <div class="mode-row"><span>Outputs</span><strong>Top-3 + all classes</strong></div>
                </div>
            </div>
            <div class="dashboard-card">
                <div class="card-title">Fast Mode</div>
                <div class="card-text">Runs cleaned-data ConvNeXt-Tiny only. This is the practical default for live webcam use.</div>
            </div>
            <div class="dashboard-card">
                <div class="card-title">Best Mode</div>
                <div class="card-text">Averages available ConvNeXt, ResNet-18, and CLIP softmax probabilities for the strongest saved ensemble.</div>
            </div>
            <div class="dashboard-card">
                <div class="card-title">Model Files</div>
                <div class="card-text">All checkpoints load locally from fer_streamlit_app/checkpoints. No training runs inside the app.</div>
            </div>
            <div class="dashboard-card">
                <div class="card-title">Model Trace</div>
                <div class="card-text">The collapsed trace shows loaded models, class probabilities, checkpoint status, and the exact crop sent to the model.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def model_mode_selector(default_fast: bool = False, key_prefix: str = "default") -> str:
    options = ["Fast Mode", "Best Mode"]
    default_index = 0 if default_fast else 1
    return st.radio(
        "Model mode",
        options,
        index=default_index,
        horizontal=True,
        key=f"{key_prefix}_model_mode",
        help="Fast Mode uses ConvNeXt only. Best Mode averages the available ensemble models.",
    )


def show_model_messages(bundle: ModelBundle) -> None:
    if not bundle.models:
        st.error("No model checkpoint could be loaded. Check the files inside fer_streamlit_app/checkpoints/.")
    for warning in bundle.warnings:
        st.warning(warning)
    if bundle.models:
        loaded_names = ", ".join(model.name for model in bundle.models)
        st.caption(f"Loaded models: {loaded_names}")
    if bundle.load_reports:
        risky_reports = [
            report
            for report in bundle.load_reports
            if report.missing_keys_count > 0 or report.unexpected_keys_count > 0
        ]
        for report in risky_reports:
            if report.strict:
                st.warning(
                    f"{report.model_name} loaded with missing/unexpected keys. "
                    f"Missing: {report.missing_keys_count}, unexpected: {report.unexpected_keys_count}."
                )
            elif report.missing_keys_count > 100 and "Expected partial" not in report.note:
                st.warning(
                    f"{report.model_name} has many missing keys after partial loading. "
                    "This checkpoint may not match the model architecture."
                )


def checkpoint_debug_table(bundle: ModelBundle) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": report.model_name,
                "checkpoint": report.checkpoint_filename,
                "loaded": report.loaded,
                "state_key": report.state_key,
                "strict": report.strict,
                "missing_keys": report.missing_keys_count,
                "unexpected_keys": report.unexpected_keys_count,
                "note": report.note,
            }
            for report in bundle.load_reports
        ]
    )


def probability_figure(probabilities: dict[str, float]):
    labels = list(probabilities.keys())
    values = [probabilities[label] * 100 for label in labels]
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    fig.patch.set_facecolor("#04070D")
    ax.set_facecolor("#04070D")
    bars = ax.barh(labels, values, color="#22D3EE", alpha=0.72)
    ax.set_xlim(0, 100)
    ax.invert_yaxis()
    ax.set_xlabel("Probability (%)", color="#CBD5E1")
    ax.tick_params(colors="#CBD5E1")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(axis="x", color="#94A3B8", alpha=0.10)
    for bar, value in zip(bars, values):
        ax.text(value + 1, bar.get_y() + bar.get_height() / 2, f"{value:.1f}%", va="center", color="#F8FAFC", fontsize=9)
    plt.tight_layout()
    return fig


def result_panel(result: dict) -> None:
    mode = result.get("mode", "Selected mode")
    st.markdown(
        f"""
        <div class="result-card">
            <div class="card-text">Prediction</div>
            <div class="result-label">{result["label"]}</div>
            <div class="confidence-pill">{result["confidence"] * 100:.1f}% confidence</div>
            <div class="card-text" style="margin-top:0.8rem;">Mode used: {mode}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Top predictions")
    for label, confidence in result["top3"]:
        st.markdown(
            f"""
            <div class="mini-card">
                <strong>{label}</strong>
                <span style="float:right;color:#BFF9FF;">{confidence * 100:.1f}%</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("#### Class probabilities")
    st.pyplot(probability_figure(result["probabilities"]), clear_figure=True)

    for warning in result.get("warnings", []):
        st.warning(warning)


def prediction_debug_panel(
    result: dict,
    bundle: ModelBundle,
    crop_image: Image.Image,
    key_prefix: str,
) -> None:
    with st.expander("Model Trace", expanded=False):
        st.markdown("**Checkpoint loading**")
        if bundle.load_reports:
            st.dataframe(checkpoint_debug_table(bundle), use_container_width=True, hide_index=True)
        else:
            st.info("No checkpoint load reports are available.")

        st.markdown("**Models used**")
        st.write(result.get("model_names", []))
        st.markdown("**Ensemble weights**")
        st.write(result.get("ensemble_weights", {}))

        st.markdown("**Face detection**")
        st.write(
            {
                "face_found": result.get("face_detected"),
                "bbox": result.get("face_box"),
                "original_image_size": result.get("original_size"),
                "model_input_crop_size": result.get("crop_size"),
                "model_mode": result.get("mode"),
            }
        )

        st.markdown("**Class index mapping**")
        st.dataframe(pd.DataFrame(result["class_index_mapping"]), use_container_width=True, hide_index=True)

        st.markdown("**Per-model probabilities**")
        for model_name, payload in result["per_model_probabilities"].items():
            st.caption(
                f"{model_name} | {payload['checkpoint']} | "
                f"weight={payload.get('ensemble_weight', 1.0)} | tta={payload.get('tta', 'none')}"
            )
            model_rows = [
                {
                    "class_index": idx,
                    "class_label": label,
                    "probability": probability,
                }
                for idx, (label, probability) in enumerate(payload["probabilities"].items())
            ]
            st.dataframe(pd.DataFrame(model_rows), use_container_width=True, hide_index=True)
            st.code(np.array(payload["probability_vector"]).__repr__(), language="python")

        st.markdown("**Final averaged probabilities**")
        st.dataframe(pd.DataFrame(result["sorted_probabilities"]), use_container_width=True, hide_index=True)
        st.code(np.array(result["raw_final_probability_vector"]).__repr__(), language="python")

        show_stats = st.checkbox(
            "Show preprocessing tensor stats",
            key=f"{key_prefix}_tensor_stats_{result.get('mode', 'mode').replace(' ', '_').lower()}",
        )
        if show_stats:
            st.markdown("**Preprocessing tensor stats for the exact model input crop**")
            st.dataframe(
                pd.DataFrame(preprocessing_tensor_stats(crop_image, bundle)),
                use_container_width=True,
                hide_index=True,
            )


def load_image_from_upload(uploaded_file) -> Image.Image:
    image = Image.open(BytesIO(uploaded_file.getvalue()))
    return ImageOps.exif_transpose(image).convert("RGB")


def prediction_workflow(image: Image.Image, mode: str, key_prefix: str) -> None:
    left, right = st.columns([1.16, 0.84], gap="large")
    image = ImageOps.exif_transpose(image).convert("RGB")
    face_image, face_found, face_box = detect_and_crop_face(image)
    boxed_image = draw_face_box(image, face_box)

    with left:
        st.markdown(
            """
            <div class="premium-card">
                <div class="card-title">Input</div>
                <div class="card-text">The app detects the largest face and sends the displayed crop to the model. If no face is found, it falls back to the full image.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        preview_col, crop_col = st.columns([0.62, 0.38], gap="large")
        with preview_col:
            st.image(
                boxed_image,
                caption="Original input image with detected face box",
                width=520,
            )
        with crop_col:
            st.image(
                face_image,
                caption="This exact crop was sent to the model.",
                width=320,
            )
        if not face_found:
            st.warning("No face detected; using full image.")
        else:
            st.caption(f"Largest detected face selected. BBox: {face_box}")

    with right:
        st.markdown(
            """
            <div class="premium-card">
                <div class="card-title">Prediction</div>
                <div class="card-text">Emotion label, confidence, top predictions, and full class probabilities.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.spinner(f"Loading {mode} and running inference..."):
            bundle = cached_model_bundle(mode)
            show_model_messages(bundle)
            if bundle.models:
                try:
                    result = debug_predict_image(
                        image,
                        mode=mode,
                        bundle=bundle,
                        face_image=face_image,
                        face_found=face_found,
                        face_box=face_box,
                    )
                    result_panel(result)
                    prediction_debug_panel(result, bundle, face_image, key_prefix=key_prefix)
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")


def image_upload_tab() -> None:
    mode = model_mode_selector(default_fast=False, key_prefix="image_upload")
    uploaded_file = st.file_uploader(
        "Upload a face image",
        type=["jpg", "jpeg", "png"],
        key="image_file_upload",
    )
    if uploaded_file is not None:
        image = load_image_from_upload(uploaded_file)
        prediction_workflow(image, mode, key_prefix="image_upload")
    else:
        st.info("Upload a JPG or PNG image to start.")


def webcam_photo_tab() -> None:
    mode = model_mode_selector(default_fast=False, key_prefix="webcam_photo")
    st.markdown('<div class="camera-helper-label">Take a webcam photo</div>', unsafe_allow_html=True)
    camera_left, camera_center, camera_right = st.columns([0.08, 0.84, 0.08])
    with camera_center:
        camera_file = st.camera_input(
            "Take a webcam photo",
            key="webcam_photo_input",
            label_visibility="collapsed",
        )
    if camera_file is not None:
        image = load_image_from_upload(camera_file)
        prediction_workflow(image, mode, key_prefix="webcam_photo")
    else:
        st.info("Allow camera access and take a photo to run the demo.")


class EmotionVideoProcessor(VideoProcessorBase):
    def __init__(self, bundle: ModelBundle) -> None:
        self.bundle = bundle
        self.lock = threading.Lock()
        self.last_prediction = "Waiting"
        self.last_confidence = 0.0
        self.last_box = None
        self.last_time = 0.0

    def recv(self, frame):
        image_bgr = frame.to_ndarray(format="bgr24")
        now = time.time()

        if now - self.last_time > 0.45:
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            try:
                result = debug_predict_image(pil_image, mode=self.bundle.mode, bundle=self.bundle)
                with self.lock:
                    self.last_prediction = result["label"]
                    self.last_confidence = result["confidence"]
                    self.last_box = result["face_box"]
                    self.last_time = now
            except Exception:
                with self.lock:
                    self.last_prediction = "Unavailable"
                    self.last_confidence = 0.0
                    self.last_box = None
                    self.last_time = now

        with self.lock:
            label = self.last_prediction
            confidence = self.last_confidence
            box = self.last_box

        if box is not None:
            x1, y1, x2, y2 = box
            cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (165, 180, 252), 2)

        overlay = f"{label}  {confidence * 100:.1f}%"
        cv2.rectangle(image_bgr, (18, 18), (18 + max(260, len(overlay) * 14), 64), (0, 0, 0), -1)
        cv2.putText(image_bgr, overlay, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (245, 245, 247), 2, cv2.LINE_AA)
        return av.VideoFrame.from_ndarray(image_bgr, format="bgr24")


def live_camera_tab() -> None:
    mode = model_mode_selector(default_fast=True, key_prefix="live_camera")
    st.caption("Fast Mode is recommended for smoother live inference.")
    if mode == "Best Mode":
        st.warning("Best Mode uses multiple models and may run slower during live webcam inference.")

    if not WEBRTC_AVAILABLE:
        st.error("Live Camera needs streamlit-webrtc and its runtime dependencies.")
        st.warning(f"Import error: {WEBRTC_IMPORT_ERROR}")
        st.info("Run `pip install -r requirements.txt`, then restart Streamlit. You can use the fallback camera capture below for now.")
        fallback_file = st.camera_input("Fallback webcam photo", key="live_camera_fallback_input")
        if fallback_file is not None:
            image = load_image_from_upload(fallback_file)
            prediction_workflow(image, mode, key_prefix="live_camera_fallback")
        return

    with st.spinner(f"Loading {mode} for live camera..."):
        bundle = cached_model_bundle(mode)
    show_model_messages(bundle)
    if not bundle.models:
        return

    def processor_factory():
        return EmotionVideoProcessor(bundle)

    try:
        webrtc_streamer(
            key=f"fer_live_{mode.lower().replace(' ', '_')}",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=processor_factory,
            media_stream_constraints={
                "video": {
                    "width": {"ideal": 960},
                    "height": {"ideal": 540},
                    "frameRate": {"ideal": 24, "max": 30},
                },
                "audio": False,
            },
            video_html_attrs={
                "autoPlay": True,
                "controls": True,
                "playsInline": True,
                "muted": True,
                "style": {
                    "width": "100%",
                    "maxWidth": "720px",
                    "maxHeight": "470px",
                    "display": "block",
                    "margin": "0 auto",
                    "borderRadius": "24px",
                    "objectFit": "contain",
                    "backgroundColor": "#050509",
                },
            },
            async_processing=True,
        )
    except Exception as exc:
        st.warning(f"Live camera could not start: {exc}")


def model_info_tab() -> None:
    st.markdown(
        """
        <div class="overview-grid">
            <div class="overview-panel">
                <div class="card-title">Project details</div>
                <p><strong>Dataset:</strong> FER-2013</p>
                <p><strong>Classes:</strong> Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise</p>
                <p><strong>Best test accuracy:</strong> 70.23%</p>
                <p>The app runs inference only. No training happens inside Streamlit.</p>
            </div>
            <div class="overview-panel">
                <div class="card-title">Runtime modes</div>
                <div class="mode-list">
                    <div class="mode-row"><span>Fast Mode</span><strong>ConvNeXt-Tiny</strong></div>
                    <div class="mode-row"><span>Best Mode</span><strong>CLIP + cleaned ensemble</strong></div>
                    <div class="mode-row"><span>Live default</span><strong>Fast Mode</strong></div>
                </div>
                <p>Predictions depend on lighting, pose, face crop quality, and how close the input is to FER-2013.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.code(str(CHECKPOINT_DIR), language="text")

    expected = pd.DataFrame(
        {
            "Checkpoint": [
                "convnext_tiny_cleaned_best_weighted_f1.pth",
                "resnet18_cleaned_best_weighted_f1.pth",
                "resnet18_finetune_best.pth",
                "clip_finetune_best.pth",
                "clip_linear_probe_best.pth",
            ],
            "Purpose": [
                "Fast Mode and ensemble CNN branch",
                "Preferred ResNet branch for Best Mode",
                "Fallback ResNet branch",
                "CLIP branch for Best Mode",
                "Reference linear-probe checkpoint",
            ],
        }
    )
    st.dataframe(expected, use_container_width=True, hide_index=True)
    st.caption("The app only loads saved checkpoints. It does not train models.")


def recognition_demo_section() -> None:
    st.markdown('<div class="section-title">Recognition Demo</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Start here. Choose an image upload, webcam snapshot, or live camera stream and the app will show the exact model input crop plus prediction details.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="demo-shell">
            <div class="demo-intro">
                <div>
                    <div class="demo-kicker">Model interface</div>
                    <div class="demo-title">Run emotion inference</div>
                    <div>Use Fast Mode for smoother webcam interaction or Best Mode for the saved ensemble.</div>
                </div>
                <div class="demo-note">The prediction panel shows confidence, top classes, probabilities, and checkpoint trace details.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    upload_tab, webcam_tab, live_tab, info_tab = st.tabs(
        ["Image Upload", "Webcam Photo", "Live Camera", "Model Info"]
    )
    with upload_tab:
        image_upload_tab()
    with webcam_tab:
        webcam_photo_tab()
    with live_tab:
        live_camera_tab()
    with info_tab:
        model_info_tab()


def footer() -> None:
    st.markdown(
        """
        <div class="footer">
            CSE 676 Deep Learning | FER-2013 | University at Buffalo<br>
            Built with PyTorch, Streamlit, CLIP, and ensemble deep learning models.
        </div>
        """,
        unsafe_allow_html=True,
    )


def sidebar_panel() -> None:
    with st.sidebar:
        st.markdown("### Demo controls")
        st.caption("Choose Fast Mode or Best Mode inside each demo tab before running inference.")
        st.markdown(
            """
            **Fast Mode**  
            Cleaned-data ConvNeXt-Tiny. Best for live webcam speed.

            **Best Mode**  
            Uses the available saved ensemble with CLIP when it loads successfully.

            **Input handling**  
            The app predicts from the detected face crop. If no face is found, it falls back to the full image.
            """
        )


def main() -> None:
    apply_styles()
    sidebar_panel()
    hero_section()
    recognition_demo_section()
    project_story_section()
    performance_section()
    model_dashboard_section()
    footer()


if __name__ == "__main__":
    main()
