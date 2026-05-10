# Facial Emotion Recognition

This project builds an end-to-end facial emotion recognition system using FER-2013. We compare a baseline CNN, ResNet-18 transfer learning, EfficientNet-B0, ConvNeXt-Tiny, CLIP foundation-model experiments, cleaned-data training, test-time augmentation, and soft-voting ensembles.

## Dataset

FER-2013 contains low-resolution 48x48 grayscale face images across seven classes:

`Angry`, `Disgust`, `Fear`, `Happy`, `Neutral`, `Sad`, and `Surprise`.

The best final result was the **CLIP + Cleaned-data Ensemble**, which achieved about **70.23% test accuracy** on the FER-2013 test set.

## Main Components

- Checkpoint 2 notebook
- Final training and comparison notebook
- CLIP foundation-model notebook
- Streamlit demo with image upload, webcam photo, and live webcam inference
- Final NeurIPS-style report

## Repository Structure

```text
EMOTION_PROJECT/
├── notebooks/
│   ├── checkpoint2_emotion_recognition.ipynb
│   ├── final_training_and_comparison.ipynb
│   └── clip_foundation_model.ipynb
├── fer_streamlit_app/
│   ├── app.py
│   ├── model_utils.py
│   ├── requirements.txt
│   ├── README.md
│   └── checkpoints/
├── final_neurips_report/
│   ├── final_report.tex
│   ├── references.bib
│   ├── final_report.pdf
│   └── figures/
├── reports/
├── figures/
├── docs/
├── requirements.txt
└── README.md
```

- `notebooks/`: training, comparison, and CLIP experiment notebooks.
- `fer_streamlit_app/`: saved-checkpoint inference demo.
- `reports/`: final report PDF.
- `final_neurips_report/`: LaTeX source, references, figures, and compiled PDF.
- `figures/`: selected report figures for quick viewing.
- `docs/`: course/project requirement document.

## Notebooks

The main notebooks are:

- `notebooks/checkpoint2_emotion_recognition.ipynb`
- `notebooks/final_training_and_comparison.ipynb`
- `notebooks/clip_foundation_model.ipynb`

Training was done on Google Colab Pro GPU. Install the shared dependencies with:

```bash
pip install -r requirements.txt
```

Do not run all notebooks from top to bottom unless the FER-2013 dataset path and GPU runtime are configured.

## Streamlit Demo

The Streamlit app loads saved checkpoints and performs inference only.

```bash
cd fer_streamlit_app
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Checkpoint files must be downloaded separately from:

```text
https://drive.google.com/file/d/1JhPvqTwidv3WgYBOuxZ6g2JPqzIM-V14/view?usp=sharing
```

Place them in:

```text
fer_streamlit_app/checkpoints/
```

The required checkpoint names are listed in `fer_streamlit_app/checkpoints/README.md`.

## Final Report

The final NeurIPS-style report is in:

```text
final_neurips_report/final_report.pdf
```

## Team

- Anirudh Ramesh
- Rohan Marar

Course: CSE 676 Deep Learning, Spring 2026

GitHub Repository: https://github.com/Anir11/Facial-Emotion-Recognition
