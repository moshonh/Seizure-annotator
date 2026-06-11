# NeuroAnnotate
## Ictal Semiology Annotation Tool
**Rambam Medical Center — Epilepsy Service**

Based on: Beniczky et al., *ILAE Glossary of Seizure Semiology*, Epileptic Disorders 2022; 24(3):447-495

---

## Setup (Windows)

### 1. Install Python 3.10+
Download from https://python.org — check "Add to PATH" during install.

### 2. Open Command Prompt and install dependencies

```cmd
cd seizure_annotator
pip install -r requirements.txt
```

> **Note**: The package uses `opencv-python-headless` (no GUI dependencies) which is compatible with all environments including Streamlit Cloud and Python 3.14+.

### 3. Run the app

```cmd
streamlit run app.py
```

The browser opens automatically at http://localhost:8501

---

## Usage

### Workflow for the student

1. **Watch the seizure** in the Natus/Persyst system
2. **Open NeuroAnnotate** in the browser
3. Fill in **Patient ID**, **Seizure #**, and **Annotator initials** in the sidebar
4. **Upload the AVI file** — the video loads in the left panel
5. Use the **video controls** to navigate:
   - Frame-by-frame (◀ -1 / +1 ▶)
   - Jump 10 frames (◀ -10 / +10 ▶)
   - Timeline scrubber
6. **Mark ictal phases** by clicking at the right frame:
   - Pre-ictal → Ictal onset → Mid-ictal → Post-ictal
7. Fill in the **ILAE Semiology Form** on the right:
   - Tick checkboxes for observed features
   - For lateralizable signs, select Left/Right/Bilateral
   - Add free-text clinical description in ILAE notation
8. Click **✓ Add to Training Set**

### Gaze Overlay (AI assist)

Toggle **Gaze estimation** in the sidebar. On first use it downloads the gazelle-dinov3 model (~200MB). Subsequent uses are instant.

The overlay shows:
- 🟢 Green circle = gaze point (IN frame)
- 🟠 Orange circle = gaze OUT of frame (aversion)
- Heatmap = confidence distribution

### Saving Training Data

- Records accumulate in memory as you annotate
- When ready (≥10 cases recommended), click **💾 Save to Training**
- JSON file saved to `~/NeuroAnnotate_Training/training_YYYYMMDD_HHMMSS.json`

---

## JSON Output Format

Each record includes:

```json
{
  "record_id": "RMB-2024-0042_sz01_20240601120000",
  "metadata": { "patient_id", "annotator", "video_filename", ... },
  "diagnosis": { "seizure_type": "ES", "localization": "Temporal mesial", ... },
  "ictal_phases": { "ictal_onset": {"frame": 450, "time_sec": 18.0}, ... },
  "ictal_duration_sec": 62.5,
  "semiology": {
    "consciousness": { "awareness_impaired": true },
    "elementary_motor": { "dystonic_posturing": true, "dystonic_posturing__lat": "Right" },
    "complex_motor": { "automatisms_oral": true },
    "oculomotor_gaze": { "gaze_deviation": true, "gaze_deviation__lat": "Left" },
    "pnes_features": {},
    "autonomic": { "auto_epigastric": true },
    "postictal": { "postictal_nose_wipe": true, "postictal_nose_wipe__lat": "Left" }
  },
  "lateralization_map": { "dystonic_posturing": "Right", "gaze_deviation": "Left" },
  "clinical_notes": "Epigastric aura → L gaze deviation + R arm dystonia → oro-alimentary automatisms → L nose wiping",
  "ml_features": {
    "pnes_feature_count": 0,
    "es_feature_count": 6,
    "temporal_feature_count": 5,
    "has_lateralizing_sign": true,
    "has_pnes_marker": false
  }
}
```

---

## ILAE Semiology Sections

| Section | Key Features |
|---------|-------------|
| Consciousness | Awareness, Responsiveness |
| Elementary Motor | Tonic, Clonic, Versive, Figure-of-4, Fencing, Dystonia |
| Complex Motor | Automatisms (oral/manual/genital), Hyperkinetic |
| Oculomotor | Gaze deviation, Eye closure, Flutter |
| PNES Features | Gradual onset, Pelvic thrust, Eye closure, Asynchronous movements |
| Autonomic | Epigastric aura, Tachycardia, Apnea, Hypersalivation |
| Sensory/Cognitive | Auras, Aphasia, Déjà vu, Fear |
| Postictal | Todd's paresis, Nose wiping, Aphasia, Hyperpnea |

---

## Notes for the PI

- All data stays **local** — no cloud upload
- JSON files are human-readable and ML-ready
- The `ml_features` field pre-computes binary feature vectors for classifier input
- Patient IDs are stored as-is — de-identify before sharing externally
