from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageEnhance, ImageOps
from torchvision import models, transforms

try:
    import open_clip
    OPEN_CLIP_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - surfaced in the UI at runtime
    open_clip = None
    OPEN_CLIP_IMPORT_ERROR = exc


BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = BASE_DIR / "checkpoints"

MODEL_CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
DISPLAY_CLASS_NAMES = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

CONVNEXT_CHECKPOINT = CHECKPOINT_DIR / "convnext_tiny_cleaned_best_weighted_f1.pth"
RESNET_CLEANED_CHECKPOINT = CHECKPOINT_DIR / "resnet18_cleaned_best_weighted_f1.pth"
RESNET_FALLBACK_CHECKPOINT = CHECKPOINT_DIR / "resnet18_finetune_best.pth"
CLIP_FINETUNE_CHECKPOINT = CHECKPOINT_DIR / "clip_finetune_best.pth"
CLIP_LINEAR_CHECKPOINT = CHECKPOINT_DIR / "clip_linear_probe_best.pth"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


@dataclass
class LoadReport:
    model_name: str
    checkpoint_filename: str
    checkpoint_path: str
    loaded: bool
    state_key: str
    strict: bool
    missing_keys_count: int = 0
    unexpected_keys_count: int = 0
    missing_keys: list[str] | None = None
    unexpected_keys: list[str] | None = None
    note: str = ""


@dataclass
class LoadedModel:
    name: str
    model: nn.Module
    transform_name: str
    checkpoint_path: Path
    tta: str = "none"
    ensemble_weight: float = 1.0
    load_report: LoadReport | None = None


@dataclass
class ModelBundle:
    mode: str
    device: torch.device
    models: list[LoadedModel]
    warnings: list[str]
    load_reports: list[LoadReport]


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def imagenet_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def clip_transform() -> transforms.Compose:
    interpolation = getattr(transforms, "InterpolationMode", None)
    interpolation = interpolation.BICUBIC if interpolation else Image.BICUBIC
    return transforms.Compose(
        [
            transforms.Resize((224, 224), interpolation=interpolation),
            transforms.ToTensor(),
            transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ]
    )


def create_convnext_tiny(num_classes: int = 7, dropout: float = 0.45) -> nn.Module:
    model = models.convnext_tiny(weights=None)
    in_features = model.classifier[2].in_features
    model.classifier = nn.Sequential(
        model.classifier[0],
        model.classifier[1],
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )
    return model


def create_resnet18(num_classes: int = 7) -> nn.Module:
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, num_classes),
    )
    return model


class CLIPEmotionClassifier(nn.Module):
    def __init__(
        self,
        clip_model: nn.Module,
        feature_dim: int,
        num_classes: int = 7,
        dropout: float = 0.30,
    ) -> None:
        super().__init__()
        self.clip_model = clip_model
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.clip_model.encode_image(images)
        return self.classifier(features.float())


def safe_torch_load(path: Path, device: torch.device) -> Any:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def extract_state_dict(checkpoint: Any) -> tuple[dict[str, torch.Tensor], str]:
    if isinstance(checkpoint, dict):
        for key in ["model_state_dict", "state_dict", "trainable_model_state_dict"]:
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value, key
        if all(hasattr(v, "shape") for v in checkpoint.values()):
            return checkpoint, "raw_state_dict"
    return checkpoint, "unknown"


def strip_common_prefixes(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    state = dict(state_dict)
    for prefix in ["module.", "model.", "net."]:
        if state and all(key.startswith(prefix) for key in state):
            state = {key[len(prefix) :]: value for key, value in state.items()}
    return state


def _short_keys(keys: list[str], limit: int = 20) -> list[str]:
    return list(keys[:limit])


def class_order_from_checkpoint(checkpoint: Any) -> list[str] | None:
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("class_names"), list):
        return [str(name).lower() for name in checkpoint["class_names"]]
    return None


def validate_class_order(checkpoint: Any, model_name: str) -> list[str]:
    checkpoint_classes = class_order_from_checkpoint(checkpoint)
    if checkpoint_classes is None:
        return [f"{model_name} checkpoint does not include class_names; using default ImageFolder alphabetical order."]
    if checkpoint_classes != MODEL_CLASS_NAMES:
        return [
            f"{model_name} checkpoint class order is {checkpoint_classes}, but app uses {MODEL_CLASS_NAMES}."
        ]
    return []


def load_state_dict_checked(
    model: nn.Module,
    checkpoint: Any,
    checkpoint_path: Path,
    model_name: str,
    strict: bool = True,
    expected_partial: bool = False,
) -> LoadReport:
    state, state_key = extract_state_dict(checkpoint)
    if not isinstance(state, dict):
        raise RuntimeError("Checkpoint does not contain a state_dict.")

    state = strip_common_prefixes(state)
    result = model.load_state_dict(state, strict=strict)
    missing = list(getattr(result, "missing_keys", []))
    unexpected = list(getattr(result, "unexpected_keys", []))
    note = "Loaded with strict validation."
    if not strict:
        note = "Partial checkpoint load; missing/unexpected keys are reported below."
        if expected_partial:
            note = "Expected partial checkpoint load; frozen base weights come from the pretrained model."

    return LoadReport(
        model_name=model_name,
        checkpoint_filename=checkpoint_path.name,
        checkpoint_path=str(checkpoint_path),
        loaded=True,
        state_key=state_key,
        strict=strict,
        missing_keys_count=len(missing),
        unexpected_keys_count=len(unexpected),
        missing_keys=_short_keys(missing),
        unexpected_keys=_short_keys(unexpected),
        note=note,
    )


def load_convnext_cleaned(device: torch.device) -> tuple[LoadedModel | None, list[str]]:
    warnings: list[str] = []
    if not CONVNEXT_CHECKPOINT.exists():
        return None, [f"Missing ConvNeXt checkpoint: {CONVNEXT_CHECKPOINT.name}"]

    model = create_convnext_tiny(dropout=0.45).to(device)
    checkpoint = safe_torch_load(CONVNEXT_CHECKPOINT, device)
    warnings.extend(validate_class_order(checkpoint, "ConvNeXt"))
    report = load_state_dict_checked(
        model,
        checkpoint,
        CONVNEXT_CHECKPOINT,
        "Cleaned-data ConvNeXt-Tiny",
        strict=True,
    )
    model.eval()
    return (
        LoadedModel(
            name="Cleaned-data ConvNeXt-Tiny",
            model=model,
            transform_name="imagenet",
            checkpoint_path=CONVNEXT_CHECKPOINT,
            tta="convnext",
            ensemble_weight=1.5,
            load_report=report,
        ),
        warnings,
    )


def load_resnet18_best(device: torch.device) -> tuple[LoadedModel | None, list[str]]:
    warnings: list[str] = []
    candidates = [
        ("Cleaned-data ResNet-18", RESNET_CLEANED_CHECKPOINT),
        ("ResNet-18 Fine-Tuning", RESNET_FALLBACK_CHECKPOINT),
    ]

    for name, path in candidates:
        if not path.exists():
            warnings.append(f"Missing {name} checkpoint: {path.name}")
            continue
        try:
            model = create_resnet18().to(device)
            checkpoint = safe_torch_load(path, device)
            warnings.extend(validate_class_order(checkpoint, name))
            report = load_state_dict_checked(model, checkpoint, path, name, strict=True)
            model.eval()
            if path == RESNET_FALLBACK_CHECKPOINT:
                warnings.append("Using ResNet-18 fine-tuned checkpoint as fallback.")
            return (
                LoadedModel(
                    name=name,
                    model=model,
                    transform_name="imagenet",
                    checkpoint_path=path,
                    tta="none",
                    ensemble_weight=1.5,
                    load_report=report,
                ),
                warnings,
            )
        except Exception as exc:
            warnings.append(f"Could not load {name} from {path.name}: {exc}")

    return None, warnings


def load_clip_finetuned(device: torch.device) -> tuple[LoadedModel | None, list[str]]:
    warnings: list[str] = []
    if not CLIP_FINETUNE_CHECKPOINT.exists():
        return None, [f"Missing CLIP checkpoint: {CLIP_FINETUNE_CHECKPOINT.name}"]
    if open_clip is None:
        return None, [
            "open_clip_torch is not installed or failed to import, so CLIP cannot be loaded. "
            "Run: pip install -r requirements.txt. "
            f"Import error: {OPEN_CLIP_IMPORT_ERROR}"
        ]

    checkpoint = safe_torch_load(CLIP_FINETUNE_CHECKPOINT, device)
    warnings.extend(validate_class_order(checkpoint, "CLIP"))
    clip_model_name = "ViT-B-32"
    clip_pretrained = "openai"
    feature_dim = 512
    if isinstance(checkpoint, dict):
        clip_model_name = checkpoint.get("clip_model_name", clip_model_name)
        clip_pretrained = checkpoint.get("clip_pretrained", clip_pretrained)
        feature_dim = int(checkpoint.get("feature_dim", feature_dim))

    clip_model, _, _ = open_clip.create_model_and_transforms(
        clip_model_name,
        pretrained=clip_pretrained,
    )
    model = CLIPEmotionClassifier(clip_model, feature_dim=feature_dim).to(device)

    report = load_state_dict_checked(
        model,
        checkpoint,
        CLIP_FINETUNE_CHECKPOINT,
        "CLIP Fine-Tuning",
        strict=False,
        expected_partial=True,
    )
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("classifier_state_dict"), dict):
        model.classifier.load_state_dict(checkpoint["classifier_state_dict"])
    else:
        warnings.append("CLIP checkpoint does not include classifier_state_dict; classifier came from partial model state.")

    model.eval()
    return (
        LoadedModel(
            name="CLIP Fine-Tuning",
            model=model,
            transform_name="clip",
            checkpoint_path=CLIP_FINETUNE_CHECKPOINT,
            tta="hflip",
            ensemble_weight=1.0,
            load_report=report,
        ),
        warnings,
    )


def load_model_bundle(mode: str, device: torch.device | None = None) -> ModelBundle:
    device = device or get_device()
    models_loaded: list[LoadedModel] = []
    warnings: list[str] = []
    load_reports: list[LoadReport] = []

    try:
        convnext_model, convnext_warnings = load_convnext_cleaned(device)
        warnings.extend(convnext_warnings)
        if convnext_model is not None:
            if mode == "Fast Mode":
                convnext_model.tta = "none"
                convnext_model.ensemble_weight = 1.0
            models_loaded.append(convnext_model)
            if convnext_model.load_report is not None:
                load_reports.append(convnext_model.load_report)
    except Exception as exc:
        warnings.append(f"Could not load ConvNeXt checkpoint: {exc}")

    if mode == "Best Mode":
        resnet_model = None
        clip_model = None
        try:
            resnet_model, resnet_warnings = load_resnet18_best(device)
            warnings.extend(resnet_warnings)
        except Exception as exc:
            warnings.append(f"Could not load ResNet-18 checkpoint: {exc}")
        try:
            clip_model, clip_warnings = load_clip_finetuned(device)
            warnings.extend(clip_warnings)
        except Exception as exc:
            warnings.append(f"Could not load CLIP checkpoint: {exc}")
        if resnet_model is not None:
            models_loaded.append(resnet_model)
            if resnet_model.load_report is not None:
                load_reports.append(resnet_model.load_report)
        if clip_model is not None:
            models_loaded.append(clip_model)
            if clip_model.load_report is not None:
                load_reports.append(clip_model.load_report)
        else:
            warnings.append("Best Mode is running without CLIP because CLIP failed to load.")

    return ModelBundle(mode=mode, device=device, models=models_loaded, warnings=warnings, load_reports=load_reports)


def ensure_rgb(image: Image.Image) -> Image.Image:
    return ImageOps.exif_transpose(image).convert("RGB")


def detect_and_crop_face(
    pil_image: Image.Image,
    margin: float = 0.12,
) -> tuple[Image.Image, bool, tuple[int, int, int, int] | None]:
    image = ensure_rgb(pil_image)
    rgb = np.array(image)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(48, 48),
    )

    if len(faces) == 0:
        return image, False, None

    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
    image_h, image_w = rgb.shape[:2]
    center_x = float(x) + float(w) / 2.0
    center_y = float(y) + float(h) / 2.0
    side = int(round(max(float(w), float(h)) * (1.0 + 2.0 * margin)))

    x1 = int(round(center_x - side / 2.0))
    y1 = int(round(center_y - side / 2.0))
    x2 = x1 + side
    y2 = y1 + side

    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > image_w:
        x1 -= x2 - image_w
        x2 = image_w
    if y2 > image_h:
        y1 -= y2 - image_h
        y2 = image_h

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image_w, x2)
    y2 = min(image_h, y2)
    cropped = Image.fromarray(rgb[y1:y2, x1:x2]).convert("RGB")
    return cropped, True, (x1, y1, x2, y2)


def draw_face_box(
    pil_image: Image.Image,
    bbox: tuple[int, int, int, int] | None,
) -> Image.Image:
    image = ensure_rgb(pil_image).copy()
    if bbox is None:
        return image
    draw = ImageDraw.Draw(image)
    x1, y1, x2, y2 = bbox
    width = max(3, round(min(image.size) * 0.006))
    draw.rectangle((x1, y1, x2, y2), outline=(255, 178, 160), width=width)
    return image


def _variants_for_tta(image: Image.Image, tta: str) -> list[Image.Image]:
    image = ensure_rgb(image)
    if tta == "hflip":
        return [image, ImageOps.mirror(image)]
    if tta == "convnext":
        bright = ImageEnhance.Brightness(image).enhance(1.06)
        bright = ImageEnhance.Contrast(bright).enhance(1.06)
        return [image, ImageOps.mirror(image), bright]
    return [image]


def _transform_for_name(name: str) -> transforms.Compose:
    return clip_transform() if name == "clip" else imagenet_transform()


def predict_with_loaded_model(loaded_model: LoadedModel, image: Image.Image, device: torch.device) -> np.ndarray:
    transform = _transform_for_name(loaded_model.transform_name)
    variants = _variants_for_tta(image, loaded_model.tta)
    probs_sum = None
    with torch.no_grad():
        for variant in variants:
            tensor = transform(ensure_rgb(variant)).unsqueeze(0).to(device)
            logits = loaded_model.model(tensor)
            probs = F.softmax(logits.float(), dim=1)
            probs_sum = probs if probs_sum is None else probs_sum + probs

    avg_probs = probs_sum / len(variants)
    return avg_probs.squeeze(0).detach().cpu().numpy()


def _probability_dict(probabilities: np.ndarray, display: bool = True) -> dict[str, float]:
    labels = DISPLAY_CLASS_NAMES if display else MODEL_CLASS_NAMES
    return {labels[i]: float(probabilities[i]) for i in range(len(labels))}


def predict_emotion(
    image: Image.Image,
    bundle: ModelBundle,
    face_image: Image.Image | None = None,
    face_found: bool | None = None,
    face_box: tuple[int, int, int, int] | None = None,
) -> dict[str, Any]:
    if not bundle.models:
        raise RuntimeError("No models are available for prediction.")

    image = ensure_rgb(image)
    if face_image is None:
        face_image, face_found, face_box = detect_and_crop_face(image)
    else:
        face_image = ensure_rgb(face_image)
        face_found = face_box is not None if face_found is None else face_found

    all_probs = []
    all_weights = []
    model_names = []
    per_model_probabilities = {}
    runtime_warnings = []

    for loaded_model in bundle.models:
        try:
            probs = predict_with_loaded_model(loaded_model, face_image, bundle.device)
            all_probs.append(probs)
            all_weights.append(float(loaded_model.ensemble_weight))
            model_names.append(loaded_model.name)
            per_model_probabilities[loaded_model.name] = {
                "checkpoint": loaded_model.checkpoint_path.name,
                "ensemble_weight": float(loaded_model.ensemble_weight),
                "tta": loaded_model.tta,
                "probability_vector": [float(value) for value in probs.tolist()],
                "probabilities": _probability_dict(probs),
            }
        except Exception as exc:
            runtime_warnings.append(f"{loaded_model.name} prediction failed: {exc}")

    if not all_probs:
        raise RuntimeError("All available models failed during prediction.")

    prob_stack = np.stack(all_probs, axis=0)
    weight_array = np.array(all_weights, dtype=np.float64)
    ensemble_probs = np.average(prob_stack, axis=0, weights=weight_array)
    top_indices = np.argsort(ensemble_probs)[::-1]
    sorted_probabilities = [
        {
            "rank": rank + 1,
            "class_index": int(idx),
            "class_label": DISPLAY_CLASS_NAMES[int(idx)],
            "probability": float(ensemble_probs[int(idx)]),
        }
        for rank, idx in enumerate(top_indices)
    ]
    prediction = {
        "label": DISPLAY_CLASS_NAMES[int(top_indices[0])],
        "confidence": float(ensemble_probs[int(top_indices[0])]),
        "top3": [
            (DISPLAY_CLASS_NAMES[int(idx)], float(ensemble_probs[int(idx)]))
            for idx in top_indices[:3]
        ],
        "probabilities": {
            DISPLAY_CLASS_NAMES[i]: float(ensemble_probs[i])
            for i in range(len(DISPLAY_CLASS_NAMES))
        },
        "raw_final_probability_vector": [float(value) for value in ensemble_probs.tolist()],
        "per_model_probabilities": per_model_probabilities,
        "ensemble_weights": {
            model_name: float(weight)
            for model_name, weight in zip(model_names, all_weights)
        },
        "sorted_probabilities": sorted_probabilities,
        "class_index_mapping": [
            {"class_index": i, "model_label": MODEL_CLASS_NAMES[i], "display_label": DISPLAY_CLASS_NAMES[i]}
            for i in range(len(MODEL_CLASS_NAMES))
        ],
        "face_image": face_image,
        "face_box": face_box,
        "face_detected": bool(face_found),
        "original_size": image.size,
        "crop_size": face_image.size,
        "model_names": model_names,
        "warnings": runtime_warnings,
    }
    return prediction


def normalize_mode(mode: str) -> str:
    lowered = str(mode).strip().lower()
    if lowered in {"fast", "fast mode"}:
        return "Fast Mode"
    if lowered in {"best", "best mode"}:
        return "Best Mode"
    return str(mode)


def debug_predict_image(
    image: Image.Image,
    mode: str = "fast",
    bundle: ModelBundle | None = None,
    face_image: Image.Image | None = None,
    face_found: bool | None = None,
    face_box: tuple[int, int, int, int] | None = None,
) -> dict[str, Any]:
    normalized_mode = normalize_mode(mode)
    bundle = bundle or load_model_bundle(normalized_mode)
    result = predict_emotion(
        image,
        bundle,
        face_image=face_image,
        face_found=face_found,
        face_box=face_box,
    )
    result["mode"] = normalized_mode
    result["bundle_warnings"] = list(bundle.warnings)
    result["load_reports"] = [
        {
            "model_name": report.model_name,
            "checkpoint_filename": report.checkpoint_filename,
            "loaded": report.loaded,
            "state_key": report.state_key,
            "strict": report.strict,
            "missing_keys_count": report.missing_keys_count,
            "unexpected_keys_count": report.unexpected_keys_count,
            "note": report.note,
            "missing_keys_sample": report.missing_keys or [],
            "unexpected_keys_sample": report.unexpected_keys or [],
        }
        for report in bundle.load_reports
    ]
    return result


def preprocessing_tensor_stats(image: Image.Image, bundle: ModelBundle) -> list[dict[str, Any]]:
    image = ensure_rgb(image)
    rows = []
    for loaded_model in bundle.models:
        transform = _transform_for_name(loaded_model.transform_name)
        tensor = transform(image)
        rows.append(
            {
                "model": loaded_model.name,
                "transform": loaded_model.transform_name,
                "shape": list(tensor.shape),
                "min": float(tensor.min().item()),
                "max": float(tensor.max().item()),
                "mean": float(tensor.mean().item()),
                "std": float(tensor.std().item()),
            }
        )
    return rows
