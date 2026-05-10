# FER Streamlit App

This app is a local demo for the FER-2013 emotion recognition project. It loads saved checkpoints and runs inference on image uploads, webcam photos, or live webcam frames. It does not train models.

## Run Locally

```bash
cd fer_streamlit_app
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

If PyTorch installation fails on Mac, install `torch` and `torchvision` separately using the official PyTorch instructions, then run `pip install -r requirements.txt` again.

## Checkpoints

Place the trained checkpoint files in:

```text
fer_streamlit_app/checkpoints/
```

Expected files:

- `convnext_tiny_cleaned_best_weighted_f1.pth`
- `resnet18_cleaned_best_weighted_f1.pth`
- `resnet18_finetune_best.pth`
- `clip_finetune_best.pth`
- `clip_linear_probe_best.pth`

The checkpoints are not committed to GitHub because they are large model artifacts.

## Modes

- **Fast Mode:** uses the cleaned-data ConvNeXt-Tiny checkpoint. This is recommended for live webcam inference.
- **Best Mode:** uses the CLIP + cleaned-data ensemble when all required checkpoints are available. It is more expensive because it loads multiple models.

The CLIP branch uses `open_clip_torch`. The first CLIP run may download the base CLIP weights if they are not already cached.

## Notes

- Test Image Upload first.
- Use Fast Mode first for Live Camera.
- If Live Camera reports a missing dependency, rerun `pip install -r requirements.txt` in the same environment and restart Streamlit.
- If a checkpoint is missing or incompatible, the app shows a warning instead of crashing.
- Predictions and confidence values come from the loaded models; no predictions are hard-coded.
