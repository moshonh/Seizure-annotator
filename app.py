"""
NeuroAnnotate — Ictal Semiology Annotation Tool
Rambam Medical Center Epilepsy Service
Based on: Beniczky et al., ILAE Glossary of Seizure Semiology, Epileptic Disorders 2022
"""

import streamlit as st
import cv2
import numpy as np
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
import base64
from utils.ilae_schema import ILAE_SCHEMA
from utils.json_export import build_annotation_record
from utils.gaze_overlay import GazeOverlay

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroAnnotate",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

  :root {
    --bg: #0d1117;
    --surface: #161b22;
    --surface2: #21262d;
    --border: #30363d;
    --accent: #58a6ff;
    --accent2: #3fb950;
    --warn: #f78166;
    --text: #e6edf3;
    --text2: #8b949e;
    --mono: 'IBM Plex Mono', monospace;
    --sans: 'IBM Plex Sans', sans-serif;
  }

  html, body, [data-testid="stApp"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--sans) !important;
  }

  [data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
  }

  h1, h2, h3 { font-family: var(--mono) !important; color: var(--accent) !important; letter-spacing: -0.5px; }
  h2 { color: var(--text) !important; font-size: 0.85rem !important; text-transform: uppercase; letter-spacing: 2px; border-bottom: 1px solid var(--border); padding-bottom: 6px; margin-top: 20px; }
  h3 { color: var(--text2) !important; font-size: 0.75rem !important; text-transform: uppercase; letter-spacing: 1.5px; }

  .stTextInput > div > div > input,
  .stSelectbox > div > div > div,
  .stTextArea textarea {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 6px !important;
    font-family: var(--sans) !important;
  }

  .stCheckbox label { font-size: 0.85rem !important; color: var(--text) !important; }
  .stCheckbox label span { color: var(--text) !important; }

  .stButton > button {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 6px !important;
    font-family: var(--mono) !important;
    font-size: 0.8rem !important;
    transition: all 0.15s ease;
  }
  .stButton > button:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
  }

  .save-btn > button {
    background: var(--accent2) !important;
    border-color: var(--accent2) !important;
    color: #0d1117 !important;
    font-weight: 600 !important;
    width: 100%;
    padding: 0.6rem !important;
  }

  .status-bar {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 16px;
    font-family: var(--mono);
    font-size: 0.78rem;
    color: var(--text2);
    margin-bottom: 16px;
  }
  .status-bar span { color: var(--accent); }

  .section-header {
    background: var(--surface2);
    border-left: 3px solid var(--accent);
    padding: 6px 12px;
    border-radius: 0 6px 6px 0;
    font-family: var(--mono);
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: var(--accent);
    margin: 16px 0 8px 0;
  }

  .ilae-note {
    font-size: 0.72rem;
    color: var(--text2);
    font-style: italic;
    margin-top: 2px;
    margin-left: 24px;
  }

  .frame-display {
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
    background: #000;
  }

  .gaze-active { color: var(--accent2) !important; }
  .gaze-inactive { color: var(--text2) !important; }

  div[data-testid="stExpander"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
  }

  .stSlider > div > div { background: var(--border) !important; }
  .stSlider > div > div > div { background: var(--accent) !important; }

  .record-count {
    font-family: var(--mono);
    font-size: 2rem;
    color: var(--accent2);
    font-weight: 600;
  }
</style>
""", unsafe_allow_html=True)


# ── Session state init ─────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "video_path": None,
        "cap": None,
        "total_frames": 0,
        "fps": 25,
        "current_frame": 0,
        "playing": False,
        "annotations": {},
        "saved_records": [],
        "gaze_overlay": None,
        "gaze_enabled": False,
        "frame_cache": {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ── Helpers ────────────────────────────────────────────────────────────────────
def load_video(path):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    return cap, total, fps


def get_frame(cap, frame_idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def frame_to_b64(frame_rgb):
    _, buf = cv2.imencode('.jpg', cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode()


def format_time(frame_idx, fps):
    total_sec = frame_idx / fps
    m = int(total_sec // 60)
    s = total_sec % 60
    return f"{m:02d}:{s:05.2f}"


# ── SIDEBAR: Patient info ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<h1 style='font-size:1.1rem;margin-bottom:0'>🧠 NeuroAnnotate</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:0.7rem;color:#8b949e;margin-top:0'>Rambam Epilepsy Service · ILAE 2022</p>", unsafe_allow_html=True)
    st.divider()

    st.markdown("## Patient")
    patient_id = st.text_input("Patient ID", placeholder="e.g. RMB-2024-0042", key="patient_id_input")
    seizure_num = st.number_input("Seizure #", min_value=1, max_value=50, value=1, key="seizure_num_input")
    annotator = st.text_input("Annotator initials", placeholder="e.g. MH", key="annotator_input")

    st.markdown("## Upload Video (AVI)")
    uploaded = st.file_uploader("", type=["avi", "mp4", "mkv"], label_visibility="collapsed")

    if uploaded:
        if st.session_state.video_path != uploaded.name:
            # Save temp file
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".avi")
            tmp.write(uploaded.read())
            tmp.close()
            cap, total, fps = load_video(tmp.name)
            st.session_state.video_path = uploaded.name
            st.session_state.cap = cap
            st.session_state.total_frames = total
            st.session_state.fps = fps
            st.session_state.current_frame = 0
            st.session_state.frame_cache = {}
            st.success(f"✓ Loaded: {total} frames @ {fps:.1f}fps")

    st.divider()

    # Gaze overlay toggle
    st.markdown("## AI Overlay")
    gaze_col1, gaze_col2 = st.columns([3, 1])
    with gaze_col1:
        gaze_toggle = st.toggle("Gaze estimation", value=st.session_state.gaze_enabled, key="gaze_toggle")
        st.session_state.gaze_enabled = gaze_toggle
    with gaze_col2:
        status_class = "gaze-active" if gaze_toggle else "gaze-inactive"
        st.markdown(f"<p class='{status_class}' style='font-size:0.7rem;margin-top:8px'>{'ON' if gaze_toggle else 'OFF'}</p>", unsafe_allow_html=True)

    if gaze_toggle:
        if st.session_state.gaze_overlay is None:
            with st.spinner("Loading gazelle-dinov3..."):
                try:
                    st.session_state.gaze_overlay = GazeOverlay()
                    st.success("✓ gazelle-dinov3 ready")
                except Exception as e:
                    st.warning(f"Gaze model unavailable: {e}\nRunning without overlay.")
                    st.session_state.gaze_enabled = False

    st.divider()

    # Saved records counter
    n_saved = len(st.session_state.saved_records)
    st.markdown(f"<p style='font-family:IBM Plex Mono;font-size:0.75rem;color:#8b949e'>Training records</p>", unsafe_allow_html=True)
    st.markdown(f"<p class='record-count'>{n_saved}</p>", unsafe_allow_html=True)

    if n_saved > 0:
        st.markdown(f"<p style='font-size:0.72rem;color:#8b949e'>{n_saved} seizure(s) annotated</p>", unsafe_allow_html=True)

        with st.container():
            st.markdown("<div class='save-btn'>", unsafe_allow_html=True)
            if st.button(f"💾 Save to Training ({n_saved} records)", use_container_width=True):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_path = Path.home() / "NeuroAnnotate_Training" / f"training_{timestamp}.json"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                export_data = {
                    "export_timestamp": timestamp,
                    "tool_version": "1.0",
                    "ilae_reference": "Beniczky et al., Epileptic Disorders 2022",
                    "n_records": n_saved,
                    "records": st.session_state.saved_records
                }
                with open(out_path, "w") as f:
                    json.dump(export_data, f, indent=2)
                st.success(f"✓ Saved to:\n{out_path}")
            st.markdown("</div>", unsafe_allow_html=True)

        if st.button("🗑 Clear all records"):
            st.session_state.saved_records = []
            st.rerun()


# ── MAIN AREA ──────────────────────────────────────────────────────────────────
col_video, col_form = st.columns([3, 2], gap="large")

# ── LEFT: Video player ──────────────────────────────────────────────────────────
with col_video:
    st.markdown("<h1 style='font-size:1.2rem'>Video Player</h1>", unsafe_allow_html=True)

    if st.session_state.cap is None:
        st.markdown("""
        <div style='border:1px dashed #30363d;border-radius:8px;padding:60px 20px;text-align:center;color:#8b949e;font-family:IBM Plex Mono;font-size:0.8rem'>
            Upload an AVI file in the sidebar to begin<br>
            <span style='font-size:2rem;display:block;margin-top:16px;opacity:0.3'>▶</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        cap = st.session_state.cap
        total = st.session_state.total_frames
        fps = st.session_state.fps

        # Status bar
        cur = st.session_state.current_frame
        t_cur = format_time(cur, fps)
        t_tot = format_time(total, fps)
        st.markdown(f"""
        <div class='status-bar'>
            Frame <span>{cur}</span> / {total} &nbsp;·&nbsp;
            Time <span>{t_cur}</span> / {t_tot} &nbsp;·&nbsp;
            FPS <span>{fps:.1f}</span> &nbsp;·&nbsp;
            Gaze <span>{'ON' if st.session_state.gaze_enabled else 'OFF'}</span>
        </div>
        """, unsafe_allow_html=True)

        # Frame display
        frame = get_frame(cap, cur)
        if frame is not None:
            # Apply gaze overlay if enabled
            if st.session_state.gaze_enabled and st.session_state.gaze_overlay:
                try:
                    overlay_result = st.session_state.gaze_overlay.process_frame(frame)
                    display_frame = overlay_result["frame"]
                    gaze_info = overlay_result.get("gaze_info", {})
                except Exception:
                    display_frame = frame
                    gaze_info = {}
            else:
                display_frame = frame
                gaze_info = {}

            st.image(display_frame, use_container_width=True, caption=None)

            # Gaze info panel
            if gaze_info:
                g_col1, g_col2, g_col3 = st.columns(3)
                with g_col1:
                    st.metric("Gaze X", f"{gaze_info.get('x', 0):.2f}")
                with g_col2:
                    st.metric("Gaze Y", f"{gaze_info.get('y', 0):.2f}")
                with g_col3:
                    in_frame = "IN" if gaze_info.get("in_frame", True) else "OUT"
                    st.metric("In Frame", in_frame)

        # Controls
        ctrl_cols = st.columns([1, 1, 1, 1, 1])
        with ctrl_cols[0]:
            if st.button("⏮ Start"):
                st.session_state.current_frame = 0
                st.rerun()
        with ctrl_cols[1]:
            if st.button("◀ -10"):
                st.session_state.current_frame = max(0, cur - 10)
                st.rerun()
        with ctrl_cols[2]:
            if st.button("◀ -1"):
                st.session_state.current_frame = max(0, cur - 1)
                st.rerun()
        with ctrl_cols[3]:
            if st.button("+1 ▶"):
                st.session_state.current_frame = min(total - 1, cur + 1)
                st.rerun()
        with ctrl_cols[4]:
            if st.button("+10 ▶"):
                st.session_state.current_frame = min(total - 1, cur + 10)
                st.rerun()

        # Scrubber
        new_frame = st.slider(
            "Seek", 0, max(total - 1, 1), st.session_state.current_frame,
            label_visibility="collapsed", key="scrubber"
        )
        if new_frame != st.session_state.current_frame:
            st.session_state.current_frame = new_frame
            st.rerun()

        # Ictal phase markers
        st.markdown("<div class='section-header'>Ictal Phase Markers</div>", unsafe_allow_html=True)
        phase_cols = st.columns(4)
        phase_labels = ["Pre-ictal", "Ictal onset", "Mid-ictal", "Post-ictal"]
        phase_keys = ["pre_ictal", "ictal_onset", "mid_ictal", "post_ictal"]
        for i, (lbl, key) in enumerate(zip(phase_labels, phase_keys)):
            with phase_cols[i]:
                if st.button(f"Mark\n{lbl}", key=f"mark_{key}"):
                    st.session_state.annotations[key] = cur
                val = st.session_state.annotations.get(key)
                if val is not None:
                    st.markdown(f"<p style='font-family:IBM Plex Mono;font-size:0.7rem;color:#58a6ff'>f{val} · {format_time(val,fps)}</p>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<p style='font-size:0.7rem;color:#8b949e'>—</p>", unsafe_allow_html=True)


# ── RIGHT: Annotation form ─────────────────────────────────────────────────────
with col_form:
    st.markdown("<h1 style='font-size:1.2rem'>ILAE Semiology Form</h1>", unsafe_allow_html=True)

    ann = st.session_state.annotations

    # ── Diagnosis & Localization ──
    st.markdown("<div class='section-header'>Diagnosis</div>", unsafe_allow_html=True)
    d_col1, d_col2 = st.columns(2)
    with d_col1:
        ann["diagnosis"] = st.selectbox(
            "Seizure type",
            ["— select —", "ES", "PNES", "Unknown"],
            key="dx_select"
        )
    with d_col2:
        ann["es_type"] = st.selectbox(
            "ES subtype",
            ["—", "Focal", "Generalized", "Unknown"],
            key="es_type_select",
            disabled=(ann.get("diagnosis") != "ES")
        )

    if ann.get("diagnosis") == "ES" and ann.get("es_type") == "Focal":
        ann["localization"] = st.selectbox(
            "Localization",
            ["—", "Temporal mesial", "Temporal neocortical",
             "Frontal mesial (SMA/ACC)", "Orbitofrontal",
             "Frontal lateral", "Parietal", "Occipital",
             "Insular", "Unknown"],
            key="loc_select"
        )
        ann["lateralization"] = st.selectbox(
            "Lateralization",
            ["—", "Left", "Right", "Bilateral", "Unknown"],
            key="lat_select"
        )

    # ── Render ILAE sections as expanders ──
    for section_key, section in ILAE_SCHEMA.items():
        with st.expander(f"{section['icon']} {section['label']}", expanded=section.get("expanded", False)):
            for field in section["fields"]:
                fkey = f"{section_key}__{field['key']}"

                if field["type"] == "checkbox":
                    val = st.checkbox(
                        field["label"],
                        key=f"cb_{fkey}",
                        value=ann.get(fkey, False)
                    )
                    ann[fkey] = val
                    if field.get("note"):
                        st.markdown(f"<p class='ilae-note'>{field['note']}</p>", unsafe_allow_html=True)

                    # Lateralization sub-field for motor signs
                    if val and field.get("lateralizable"):
                        sub_key = f"{fkey}__lat"
                        lat_val = st.radio(
                            f"  ↳ Laterality",
                            ["Left", "Right", "Bilateral", "Unknown"],
                            horizontal=True,
                            key=f"lat_{fkey}",
                            index=["Left","Right","Bilateral","Unknown"].index(ann.get(sub_key, "Unknown"))
                        )
                        ann[sub_key] = lat_val

                elif field["type"] == "select":
                    val = st.selectbox(
                        field["label"],
                        field["options"],
                        key=f"sel_{fkey}"
                    )
                    ann[fkey] = val

    # ── Free text ──
    st.markdown("<div class='section-header'>Notes</div>", unsafe_allow_html=True)
    ann["clinical_notes"] = st.text_area(
        "Clinical description",
        placeholder="Describe semiology in ILAE notation (use → for sequence, + for simultaneous)...",
        height=100,
        key="notes_input"
    )

    # ── Confidence ──
    ann["confidence"] = st.select_slider(
        "Annotation confidence",
        options=["Low", "Medium", "High", "Definite"],
        value=ann.get("confidence", "High"),
        key="confidence_slider"
    )

    # ── Save this seizure ──
    st.divider()
    can_save = (
        patient_id.strip() != "" and
        annotator.strip() != "" and
        ann.get("diagnosis", "— select —") != "— select —"
    )

    if not can_save:
        st.markdown("<p style='font-size:0.78rem;color:#f78166'>⚠ Fill in Patient ID, Annotator, and Diagnosis to save</p>", unsafe_allow_html=True)

    if st.button("✓ Add to Training Set", disabled=not can_save, use_container_width=True, type="primary"):
        record = build_annotation_record(
            patient_id=patient_id,
            seizure_num=seizure_num,
            annotator=annotator,
            video_name=st.session_state.video_path or "unknown",
            fps=st.session_state.fps,
            total_frames=st.session_state.total_frames,
            annotations=ann.copy()
        )
        st.session_state.saved_records.append(record)
        # Reset form
        st.session_state.annotations = {}
        st.success(f"✓ Record added. Total: {len(st.session_state.saved_records)}")
        st.rerun()

    # Preview last saved record
    if st.session_state.saved_records:
        with st.expander("Preview last saved record"):
            st.json(st.session_state.saved_records[-1])
