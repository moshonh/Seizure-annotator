"""
NeuroAnnotate — Ictal Semiology Annotation Tool
Rambam Medical Center Epilepsy Service
Based on: Beniczky et al., ILAE Glossary of Seizure Semiology, Epileptic Disorders 2022
Single-file version (no external utils dependencies)
"""

import sys
import os
import streamlit as st
import cv2
import numpy as np
import json
import tempfile
from datetime import datetime
from pathlib import Path
import base64
import urllib.request

# ══════════════════════════════════════════════════════════
# ILAE SCHEMA (Beniczky et al. 2022)
# ══════════════════════════════════════════════════════════
ILAE_SCHEMA = {

    "consciousness": {
        "label": "Consciousness & Responsiveness",
        "icon": "👁",
        "expanded": True,
        "fields": [
            {"key": "awareness_intact",    "label": "Awareness intact",          "type": "checkbox", "note": "Patient later verifies retained awareness"},
            {"key": "awareness_impaired",  "label": "Awareness impaired",        "type": "checkbox", "note": "Unable to later verify retained awareness"},
            {"key": "responsive",          "label": "Responsive to commands",    "type": "checkbox"},
            {"key": "partially_responsive","label": "Partially responsive",      "type": "checkbox", "note": "Inconsistent or prolonged reaction time"},
            {"key": "non_responsive",      "label": "Non-responsive",            "type": "checkbox"},
            {"key": "behavioral_arrest",   "label": "Behavioral / Motor arrest", "type": "checkbox"},
        ]
    },

    "elementary_motor": {
        "label": "Elementary Motor Signs",
        "icon": "⚡",
        "expanded": True,
        "fields": [
            {"key": "akinetic",       "label": "Akinetic (negative motor)",  "type": "checkbox", "lateralizable": True,  "note": "Mesial premotor / inferior frontal"},
            {"key": "atonic",         "label": "Atonic",                     "type": "checkbox", "lateralizable": True},
            {"key": "myoclonic",      "label": "Myoclonic",                  "type": "checkbox", "lateralizable": True,  "note": "Contralateral → frontal/motor cortex"},
            {"key": "clonic",         "label": "Clonic (rhythmic)",          "type": "checkbox", "lateralizable": True,  "note": "Contralateral hemisphere ~90%"},
            {"key": "myoclonic_atonic","label": "Myoclonic-atonic",          "type": "checkbox"},
            {"key": "epileptic_spasm", "label": "Epileptic spasm",           "type": "checkbox", "lateralizable": True},
            {"key": "eye_blinking",   "label": "Ictal eye blinking",         "type": "checkbox", "lateralizable": True,  "note": "IPSI when unilateral"},
            {"key": "tonic_unilateral","label": "Tonic — unilateral",        "type": "checkbox", "lateralizable": True,  "note": "Contralateral ~90%"},
            {"key": "tonic_bilateral_sym","label":"Tonic — bilateral symmetric","type":"checkbox"},
            {"key": "tonic_bilateral_asym","label":"Tonic — bilateral asymmetric","type":"checkbox"},
            {"key": "chapeau_gendarme","label":"Chapeau de gendarme (ictal pouting)","type":"checkbox", "note": "Frontal: anterior prefrontal / anterior cingulate"},
            {"key": "fencing_posture","label": "Fencing posture (M2e)",      "type": "checkbox", "lateralizable": True,  "note": "CON → supplementary motor area (mesial frontal)"},
            {"key": "tonic_clonic",   "label": "Tonic-clonic",               "type": "checkbox", "lateralizable": True},
            {"key": "figure_of_four", "label": "Figure-of-4 sign",           "type": "checkbox", "lateralizable": True,  "note": "Extended arm = CON hemisphere"},
            {"key": "asymm_clonic_ending","label":"Asymmetric clonic ending", "type": "checkbox", "lateralizable": True,  "note": "Last jerks IPSI to onset"},
            {"key": "versive",        "label": "Versive (forced sustained)",  "type": "checkbox", "lateralizable": True,  "note": "CON when followed by FBTC"},
            {"key": "head_orientation","label":"Head orientation (non-versive)","type":"checkbox", "lateralizable": True,  "note": "IPSI early in TLE"},
            {"key": "gyratory",       "label": "Gyratory",                   "type": "checkbox", "lateralizable": True,  "note": "CON when head version precedes"},
            {"key": "epileptic_nystagmus","label":"Epileptic nystagmus",     "type": "checkbox", "lateralizable": True,  "note": "Fast phase CON → occipital"},
            {"key": "ictal_paresis",  "label": "Ictal paresis",              "type": "checkbox", "lateralizable": True,  "note": "CON → motor cortex"},
            {"key": "dystonic_posturing","label":"Dystonic posturing",        "type": "checkbox", "lateralizable": True,  "note": "CON when unilateral (mainly TLE)"},
        ]
    },

    "complex_motor": {
        "label": "Complex Motor / Automatisms",
        "icon": "🤲",
        "expanded": False,
        "fields": [
            {"key": "automatisms_oral",     "label": "Oro-alimentary automatisms (chewing, lip smacking)", "type": "checkbox", "note": "Temporal mesial / insulo-opercular"},
            {"key": "automatisms_distal",   "label": "Gestural automatisms — distal (hand/finger)", "type": "checkbox", "lateralizable": True, "note": "IPSI in TLE context"},
            {"key": "automatisms_proximal", "label": "Gestural automatisms — proximal (arm/leg)", "type": "checkbox"},
            {"key": "automatisms_genital",  "label": "Genital automatisms",   "type": "checkbox", "lateralizable": True, "note": "IPSI → non-dominant temporal"},
            {"key": "automatisms_mimic_gelastic","label":"Mimic — gelastic (laughing)","type":"checkbox","note":"Hypothalamic hamartoma if isolated/clusters"},
            {"key": "automatisms_mimic_dacrystic","label":"Mimic — dacrystic (crying)","type":"checkbox"},
            {"key": "automatisms_verbal",   "label": "Verbal automatisms",    "type": "checkbox", "note": "Temporal / insulo-opercular"},
            {"key": "automatisms_vocal",    "label": "Vocal automatisms (grunts, shrieks)", "type": "checkbox"},
            {"key": "hyperkinetic",         "label": "Hyperkinetic / Hypermotor behaviour", "type": "checkbox", "note": "Frontal (esp. orbitofrontal / ACC)"},
            {"key": "ictal_grasping",       "label": "Ictal grasping",        "type": "checkbox", "note": "Anterior cingulate"},
        ]
    },

    "oculomotor": {
        "label": "Oculomotor / Gaze",
        "icon": "👀",
        "expanded": True,
        "fields": [
            {"key": "gaze_deviation",       "label": "Ictal gaze deviation",  "type": "checkbox", "lateralizable": True, "note": "CON to onset focus (frontal/temporal)"},
            {"key": "gaze_aversion",        "label": "Gaze aversion / avoidance", "type": "checkbox"},
            {"key": "eyes_open_fixed",      "label": "Eyes open, fixed stare", "type": "checkbox"},
            {"key": "eye_closure_forceful", "label": "Forceful eye closure",   "type": "checkbox", "note": "High specificity for PNES"},
            {"key": "eye_closure_flutter",  "label": "Eye flutter / eyelid myoclonia", "type": "checkbox"},
            {"key": "eye_closure_subtle",   "label": "Subtle eye closure",     "type": "checkbox"},
            {"key": "upward_gaze",          "label": "Upward gaze deviation",  "type": "checkbox"},
        ]
    },

    "pnes_features": {
        "label": "PNES Suggestive Features",
        "icon": "🔴",
        "expanded": False,
        "fields": [
            {"key": "pnes_gradual_onset",   "label": "Gradual onset (>30s ramp-up)",       "type": "checkbox"},
            {"key": "pnes_prolonged",       "label": "Prolonged duration (>5 min)",         "type": "checkbox"},
            {"key": "pnes_waxing_waning",   "label": "Waxing and waning course",            "type": "checkbox"},
            {"key": "pnes_eye_closure",     "label": "Forceful / sustained eye closure",    "type": "checkbox", "note": "Strong PNES marker"},
            {"key": "pnes_pelvic_thrust",   "label": "Pelvic thrusting",                    "type": "checkbox"},
            {"key": "pnes_opisthotonus",    "label": "Opisthotonus / arc-en-ciel",          "type": "checkbox"},
            {"key": "pnes_preserved_awareness","label":"Preserved awareness with motor activity","type":"checkbox"},
            {"key": "pnes_side_to_side",    "label": "Side-to-side head movement",         "type": "checkbox"},
            {"key": "pnes_stuttering",      "label": "Ictal stuttering / crying",           "type": "checkbox"},
            {"key": "pnes_asynchronous",    "label": "Asynchronous limb movements",        "type": "checkbox"},
        ]
    },

    "autonomic": {
        "label": "Autonomic Phenomena",
        "icon": "💓",
        "expanded": False,
        "fields": [
            {"key": "auto_tachycardia",     "label": "Tachycardia",            "type": "checkbox", "note": "Temporal more than extratemporal"},
            {"key": "auto_bradycardia",     "label": "Bradycardia / asystole", "type": "checkbox", "note": "Bilateral temporal / orbitofrontal"},
            {"key": "auto_flushing",        "label": "Flushing",               "type": "checkbox"},
            {"key": "auto_pallor",          "label": "Pallor",                 "type": "checkbox"},
            {"key": "auto_piloerection",    "label": "Piloerection",           "type": "checkbox", "note": "Temporal / amygdala"},
            {"key": "auto_hypersalivation", "label": "Hypersalivation",        "type": "checkbox", "note": "Insulo-opercular"},
            {"key": "auto_epigastric",      "label": "Epigastric aura",        "type": "checkbox", "note": "Temporal (mesial) → strong TLE marker"},
            {"key": "auto_vomiting",        "label": "Ictal vomiting",         "type": "checkbox", "note": "Right temporal / insular"},
            {"key": "auto_spitting",        "label": "Ictal spitting",         "type": "checkbox", "note": "Non-dominant temporal"},
            {"key": "auto_apnea",           "label": "Ictal apnea",            "type": "checkbox", "note": "Temporal (contralateral spread)"},
            {"key": "auto_hyperventilation","label": "Ictal hyperventilation", "type": "checkbox"},
            {"key": "auto_mydriasis",       "label": "Mydriasis",              "type": "checkbox"},
        ]
    },

    "sensory_cognitive": {
        "label": "Sensory & Cognitive Phenomena",
        "icon": "🧩",
        "expanded": False,
        "fields": [
            {"key": "aura_epigastric",  "label": "Epigastric aura",            "type": "checkbox", "note": "Temporal mesial ~98% when + automatisms"},
            {"key": "aura_olfactory",   "label": "Olfactory aura",             "type": "checkbox", "note": "Amygdala / piriform / orbitofrontal"},
            {"key": "aura_gustatory",   "label": "Gustatory aura",             "type": "checkbox", "note": "Insula / peri-rolandic"},
            {"key": "aura_auditory",    "label": "Auditory aura",              "type": "checkbox", "note": "Superior temporal gyrus"},
            {"key": "aura_visual",      "label": "Visual aura (elementary)",   "type": "checkbox", "lateralizable": True, "note": "CON when unilateral → occipital"},
            {"key": "aura_visual_complex","label":"Visual aura (complex)",      "type": "checkbox", "note": "Temporal / parieto-occipital"},
            {"key": "aura_somatosensory","label":"Somatosensory aura",         "type": "checkbox", "lateralizable": True, "note": "CON → primary somatosensory cortex"},
            {"key": "aura_vestibular",  "label": "Vestibular / dizziness",     "type": "checkbox", "note": "Parieto-perisylvian"},
            {"key": "aura_deja_vu",     "label": "Déjà vu / Jamais vu",        "type": "checkbox", "note": "Temporal mesial (amygdala/hippocampus)"},
            {"key": "aura_fear",        "label": "Ictal fear / panic",         "type": "checkbox", "note": "Temporal mesial (amygdala) / orbitofrontal"},
            {"key": "aura_ecstatic",    "label": "Ecstatic / blissful feeling","type": "checkbox", "note": "Anterior dorsal insula"},
            {"key": "ictal_aphasia",    "label": "Ictal aphasia",              "type": "checkbox", "lateralizable": True, "note": "Dominant hemisphere"},
            {"key": "forced_thinking",  "label": "Forced thinking",            "type": "checkbox", "note": "Dominant frontal"},
        ]
    },

    "postictal": {
        "label": "Postictal Phenomena",
        "icon": "🌙",
        "expanded": False,
        "fields": [
            {"key": "postictal_unresponsive",   "label": "Postictal unresponsiveness", "type": "checkbox"},
            {"key": "postictal_nose_wipe",      "label": "Nose wiping",                "type": "checkbox", "lateralizable": True, "note": "IPSI in TLE (86.5%)"},
            {"key": "postictal_aphasia",        "label": "Postictal language dysfunction","type":"checkbox","lateralizable":True,"note":"Dominant hemisphere"},
            {"key": "postictal_todd",           "label": "Todd's paresis",             "type": "checkbox", "lateralizable": True, "note": "CON to onset"},
            {"key": "postictal_blindness",      "label": "Postictal blindness",        "type": "checkbox", "lateralizable": True, "note": "CON → occipital"},
            {"key": "postictal_headache",       "label": "Postictal headache",         "type": "checkbox", "lateralizable": True, "note": "IPSI in TLE (90%)"},
            {"key": "postictal_psychiatric",    "label": "Postictal psychiatric signs", "type": "checkbox"},
            {"key": "postictal_hyperpnea",      "label": "Postictal hyperpnea",        "type": "checkbox", "note": "Supports ES over PNES"},
            {"key": "postictal_cough",          "label": "Postictal coughing",         "type": "checkbox", "note": "Temporal (with stereotyped semiology)"},
        ]
    },
}


# ══════════════════════════════════════════════════════════
# JSON EXPORT
# ══════════════════════════════════════════════════════════
def build_annotation_record(
    patient_id: str,
    seizure_num: int,
    annotator: str,
    video_name: str,
    fps: float,
    total_frames: int,
    annotations: dict
) -> dict:
    """
    Build a structured JSON record for training data export.
    Follows ILAE 2022 semiology taxonomy.
    """

    def frame_to_sec(f):
        return round(f / fps, 3) if fps > 0 else None

    # Extract phase markers
    phases = {}
    for phase_key in ["pre_ictal", "ictal_onset", "mid_ictal", "post_ictal"]:
        frame_val = annotations.get(phase_key)
        if frame_val is not None:
            phases[phase_key] = {
                "frame": frame_val,
                "time_sec": frame_to_sec(frame_val)
            }

    # Derive ictal duration
    onset_sec = phases.get("ictal_onset", {}).get("time_sec")
    offset_sec = phases.get("post_ictal", {}).get("time_sec")
    duration_sec = round(offset_sec - onset_sec, 2) if (onset_sec and offset_sec) else None

    # Collect active ILAE features
    def collect_section(prefix):
        result = {}
        for k, v in annotations.items():
            if k.startswith(f"{prefix}__"):
                field = k[len(prefix)+2:]
                if v and v not in ["—", "— select —", False]:
                    result[field] = v
        return result

    # Motor lateralization map
    lateralization_map = {}
    for k, v in annotations.items():
        if k.endswith("__lat") and v:
            motor_feature = k.replace("__lat", "").split("__")[-1]
            lateralization_map[motor_feature] = v

    record = {
        "record_id": f"{patient_id}_sz{seizure_num:02d}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "metadata": {
            "patient_id": patient_id,
            "seizure_number": seizure_num,
            "annotator": annotator,
            "annotation_timestamp": datetime.now().isoformat(),
            "video_filename": video_name,
            "video_fps": fps,
            "video_total_frames": total_frames,
            "tool_version": "NeuroAnnotate 1.0",
            "ilae_reference": "Beniczky et al., Epileptic Disorders 2022"
        },
        "diagnosis": {
            "seizure_type": annotations.get("diagnosis"),
            "es_subtype": annotations.get("es_type"),
            "localization": annotations.get("localization"),
            "lateralization": annotations.get("lateralization"),
            "annotation_confidence": annotations.get("confidence", "High")
        },
        "ictal_phases": phases,
        "ictal_duration_sec": duration_sec,
        "semiology": {
            "consciousness": collect_section("consciousness"),
            "elementary_motor": collect_section("elementary_motor"),
            "complex_motor": collect_section("complex_motor"),
            "oculomotor_gaze": collect_section("oculomotor"),
            "pnes_features": collect_section("pnes_features"),
            "autonomic": collect_section("autonomic"),
            "sensory_cognitive": collect_section("sensory_cognitive"),
            "postictal": collect_section("postictal"),
        },
        "lateralization_map": lateralization_map,
        "clinical_notes": annotations.get("clinical_notes", ""),

        # Derived feature flags for ML
        "ml_features": derive_ml_features(annotations)
    }

    return record


def derive_ml_features(ann: dict) -> dict:
    """
    Pre-computed binary feature vector for ML classifier input.
    These are the key discriminating features from ILAE literature.
    """

    def has(key):
        return bool(ann.get(key, False))

    pnes_score = sum([
        has("pnes_features__pnes_gradual_onset"),
        has("pnes_features__pnes_prolonged"),
        has("pnes_features__pnes_waxing_waning"),
        has("pnes_features__pnes_eye_closure"),
        has("pnes_features__pnes_pelvic_thrust"),
        has("pnes_features__pnes_opisthotonus"),
        has("pnes_features__pnes_side_to_side"),
        has("pnes_features__pnes_asynchronous"),
    ])

    es_score = sum([
        has("consciousness__awareness_impaired"),
        has("elementary_motor__tonic_unilateral"),
        has("elementary_motor__versive"),
        has("elementary_motor__figure_of_four"),
        has("elementary_motor__fencing_posture"),
        has("elementary_motor__dystonic_posturing"),
        has("complex_motor__automatisms_oral"),
        has("oculomotor__gaze_deviation"),
        has("autonomic__auto_epigastric"),
        has("postictal__postictal_nose_wipe"),
        has("postictal__postictal_todd"),
    ])

    temporal_features = sum([
        has("complex_motor__automatisms_oral"),
        has("complex_motor__automatisms_distal"),
        has("sensory_cognitive__aura_deja_vu"),
        has("sensory_cognitive__aura_epigastric"),
        has("sensory_cognitive__aura_fear"),
        has("autonomic__auto_epigastric"),
        has("postictal__postictal_nose_wipe"),
    ])

    frontal_features = sum([
        has("complex_motor__hyperkinetic"),
        has("elementary_motor__fencing_posture"),
        has("elementary_motor__chapeau_gendarme"),
        has("elementary_motor__tonic_bilateral_asym"),
        has("complex_motor__automatisms_vocal"),
    ])

    return {
        "pnes_feature_count": pnes_score,
        "es_feature_count": es_score,
        "temporal_feature_count": temporal_features,
        "frontal_feature_count": frontal_features,
        "has_lateralizing_sign": any([
            has("elementary_motor__versive"),
            has("elementary_motor__figure_of_four"),
            has("elementary_motor__fencing_posture"),
            has("elementary_motor__dystonic_posturing"),
            has("elementary_motor__asymm_clonic_ending"),
            has("oculomotor__gaze_deviation"),
            has("postictal__postictal_todd"),
            has("postictal__postictal_nose_wipe"),
        ]),
        "has_pnes_marker": pnes_score >= 2,
        "has_oculomotor_sign": any([
            has("oculomotor__gaze_deviation"),
            has("oculomotor__eye_closure_forceful"),
            has("oculomotor__upward_gaze"),
        ]),
        "awareness_preserved": has("consciousness__awareness_intact"),
        "awareness_impaired": has("consciousness__awareness_impaired"),
    }


# ══════════════════════════════════════════════════════════
# PATIENT DETECTION OVERLAY
# Multi-cascade: fullbody → upperbody → face → eyes
# All models built into OpenCV — zero downloads
# ══════════════════════════════════════════════════════════

class GazeOverlay:
    """
    Multi-stage patient detection using OpenCV built-in Haar cascades.
    Strategy:
      1. Full body detection  → locates patient in frame
      2. Upper body detection → narrows to torso/head region  
      3. Face detection (frontal + profile) → head location
      4. Eye detection + iris → gaze direction estimate
    Works even when face is bandaged / partially occluded.
    Zero downloads — all cascades bundled with opencv-python-headless.
    """

    def __init__(self):
        data = cv2.data.haarcascades
        self.face_f   = cv2.CascadeClassifier(data + 'haarcascade_frontalface_default.xml')
        self.face_p   = cv2.CascadeClassifier(data + 'haarcascade_profileface.xml')
        self.face_alt = cv2.CascadeClassifier(data + 'haarcascade_frontalface_alt2.xml')
        self.upper    = cv2.CascadeClassifier(data + 'haarcascade_upperbody.xml')
        self.full     = cv2.CascadeClassifier(data + 'haarcascade_fullbody.xml')
        self.eye      = cv2.CascadeClassifier(data + 'haarcascade_eye.xml')
        self.leye     = cv2.CascadeClassifier(data + 'haarcascade_lefteye_2splits.xml')
        self.reye     = cv2.CascadeClassifier(data + 'haarcascade_righteye_2splits.xml')
        self._history = {"body": None, "face": None, "gaze": 0.0}
        print("Patient detection overlay ready (OpenCV multi-cascade)")

    def _detect_any(self, gray, cascades, scale=1.1, neighbors=3, min_size=(40,40)):
        """Run multiple cascades, return largest detection."""
        best = None
        for cas in cascades:
            dets = cas.detectMultiScale(gray, scaleFactor=scale,
                                        minNeighbors=neighbors, minSize=min_size)
            for d in dets:
                if best is None or d[2]*d[3] > best[2]*best[3]:
                    best = d
        return best

    def _iris_center(self, eye_gray):
        """Locate iris/pupil via thresholding."""
        if eye_gray.size < 100:
            return None
        blur = cv2.GaussianBlur(eye_gray, (7,7), 0)
        # Adaptive threshold — works under different lighting
        _, th = cv2.threshold(blur, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) < 8:
            return None
        M = cv2.moments(c)
        if M["m00"] == 0:
            return None
        return int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])

    def process_frame(self, frame_rgb: np.ndarray) -> dict:
        h, w = frame_rgb.shape[:2]
        out = frame_rgb.copy()
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        # Equalize histogram → better detection under uneven lighting
        gray_eq = cv2.equalizeHist(gray)

        info = {
            "patient_detected": False,
            "face_detected": False,
            "n_eyes": 0,
            "gaze_direction": "unknown",
            "deviation_x": 0.0,
            "in_frame": True,
            "x": 0.5, "y": 0.5,
        }

        # ── Stage 1: locate patient body ────────────────────
        body = self._detect_any(gray_eq,
                                [self.full, self.upper],
                                scale=1.05, neighbors=2, min_size=(60,60))

        if body is not None:
            bx, by, bw, bh = body
            info["patient_detected"] = True
            info["x"] = (bx + bw/2) / w
            info["y"] = (by + bh/2) / h
            self._history["body"] = body
            # Draw body box
            cv2.rectangle(out, (bx,by), (bx+bw,by+bh), (88,200,100), 2)
            cv2.putText(out, "PATIENT", (bx, by-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (88,200,100), 1)
            # Search for face in upper 60% of body ROI
            search_y1 = by
            search_y2 = by + int(bh * 0.6)
            search_x1 = bx
            search_x2 = bx + bw
        else:
            # Fall back to full frame search
            search_y1, search_y2 = 0, h
            search_x1, search_x2 = 0, w

        # ── Stage 2: face detection in search region ─────────
        roi_gray = gray_eq[search_y1:search_y2, search_x1:search_x2]
        face = self._detect_any(roi_gray,
                                [self.face_f, self.face_alt, self.face_p],
                                scale=1.08, neighbors=3, min_size=(30,30))

        if face is not None:
            fx, fy, fw, fh = face
            # Translate back to full frame coords
            fx += search_x1; fy += search_y1
            info["face_detected"] = True
            info["patient_detected"] = True
            info["x"] = (fx + fw/2) / w
            info["y"] = (fy + fh/2) / h
            self._history["face"] = (fx, fy, fw, fh)

            cv2.rectangle(out, (fx,fy), (fx+fw,fy+fh), (255,200,50), 2)
            cv2.putText(out, "FACE", (fx, fy-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,200,50), 1)

            # ── Stage 3: eye detection ────────────────────────
            face_gray = gray[fy:fy+fh, fx:fx+fw]
            upper_face = face_gray[:int(fh*0.55), :]

            eyes = self.eye.detectMultiScale(upper_face, 1.08, 3, minSize=(15,15))
            if len(eyes) == 0:
                # Try left/right eye cascades
                leyes = self.leye.detectMultiScale(upper_face, 1.08, 2, minSize=(12,12))
                reyes = self.reye.detectMultiScale(upper_face, 1.08, 2, minSize=(12,12))
                eyes = list(leyes) + list(reyes)

            info["n_eyes"] = len(eyes)
            deviations = []

            for (ex, ey, ew, eh) in list(eyes)[:2]:
                # Draw eye box
                abs_ex, abs_ey = fx+ex, fy+ey
                cv2.rectangle(out, (abs_ex,abs_ey),
                              (abs_ex+ew,abs_ey+eh), (88,166,255), 1)
                # Iris
                eye_roi = gray[abs_ey:abs_ey+eh, abs_ex:abs_ex+ew]
                iris = self._iris_center(eye_roi)
                if iris is not None:
                    ix, iy = abs_ex+iris[0], abs_ey+iris[1]
                    cv2.circle(out, (ix,iy), 4, (255,220,50), -1)
                    dev = (iris[0]/ew - 0.5) * 2
                    deviations.append(dev)

            if deviations:
                mean_dev = float(np.mean(deviations))
                info["deviation_x"] = mean_dev
                self._history["gaze"] = mean_dev
                if mean_dev < -0.25:
                    info["gaze_direction"] = "LEFT"
                elif mean_dev > 0.25:
                    info["gaze_direction"] = "RIGHT"
                else:
                    info["gaze_direction"] = "CENTER"

        elif self._history["face"] is not None:
            # Use last known face position (smoothing)
            fx, fy, fw, fh = self._history["face"]
            cv2.rectangle(out, (fx,fy), (fx+fw,fy+fh), (255,200,50,128), 1)

        # ── HUD bar ──────────────────────────────────────────
        hud = out.copy()
        cv2.rectangle(hud, (0,0), (w,38), (0,0,0), -1)
        out = cv2.addWeighted(hud, 0.55, out, 0.45, 0)

        p_col = (88,200,100) if info["patient_detected"] else (247,129,102)
        f_col = (255,200,50) if info["face_detected"] else (150,150,150)
        e_col = (88,166,255) if info["n_eyes"]>0 else (150,150,150)
        g_col = (255,220,50)

        labels = [
            (f"PATIENT:{'✓' if info['patient_detected'] else '✗'}", 8,  p_col),
            (f"FACE:{'✓' if info['face_detected'] else '✗'}",        160, f_col),
            (f"EYES:{info['n_eyes']}",                                280, e_col),
            (f"GAZE:{info['gaze_direction']}",                        370, g_col),
            (f"DEV:{info['deviation_x']:+.2f}",                       510, g_col),
        ]
        for txt, x, col in labels:
            cv2.putText(out, txt, (x,26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 1)

        # Gaze deviation arrow
        if info["patient_detected"]:
            cx = w//2
            arrow_end = cx + int(info["deviation_x"] * 70)
            cv2.arrowedLine(out, (cx,52), (arrow_end,52), g_col, 2, tipLength=0.3)

        return {"frame": out, "gaze_info": info}


class MockGazeOverlay:
    def __init__(self):
        pass
    def process_frame(self, frame_rgb):
        return {"frame": frame_rgb, "gaze_info": {}}

# ══════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════
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
