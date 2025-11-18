from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import torch
import torch.nn as nn
import open_clip
from PIL import Image

_MODEL_CACHE: Dict[str, Any] = {}


def load_model(
    model_name: str = "ViT-B-32",
    pretrained: str = "laion2b_s34b_b79k",
    device: str = None,
    checkpoint_path: Optional[Path] = None,
):
    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    key = (
        model_name,
        str(pretrained) if pretrained else "",
        str(checkpoint_path) if checkpoint_path else "",
        device,
    )
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    if checkpoint_path:
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=checkpoint_path
        )
    else:
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.to(device)
    model.eval()
    result = (model, preprocess, tokenizer, device)
    _MODEL_CACHE[key] = result
    return result


_TEXT_FEATURES_CACHE: Dict[str, torch.Tensor] = {}


def _get_text_features(
    prompts: List[str], model, tokenizer, device: str, cache_key_prefix: str
) -> torch.Tensor:
    cache_key = cache_key_prefix + "|" + "\u0001".join(prompts)
    if cache_key in _TEXT_FEATURES_CACHE:
        return _TEXT_FEATURES_CACHE[cache_key]
    text = tokenizer(prompts)
    with torch.no_grad():
        text_features = model.encode_text(text.to(device))
        text_features /= text_features.norm(dim=-1, keepdim=True)
    _TEXT_FEATURES_CACHE[cache_key] = text_features
    return text_features


def clip_classify(
    paths: List[Path],
    prompts: List[str],
    model_name: str = "ViT-B-32",
    pretrained: str = "laion2b_s34b_b79k",
    checkpoint_path: Optional[Path] = None,
    batch_size: Optional[int] = None,
) -> List[Tuple[str, float]]:
    model, preprocess, tokenizer, device = load_model(
        model_name, pretrained, checkpoint_path=checkpoint_path
    )
    cache_prefix = "|".join(
        [
            model_name,
            str(pretrained) if pretrained else "",
            str(checkpoint_path) if checkpoint_path else "",
            device,
        ]
    )
    text_features = _get_text_features(prompts, model, tokenizer, device, cache_prefix)
    results: List[Tuple[str, float]] = []
    with torch.no_grad():
        if not batch_size or batch_size <= 0:
            batch_size = len(paths)
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start : start + batch_size]
            images = []
            for p in batch_paths:
                img = Image.open(p).convert("RGB")
                images.append(preprocess(img))
            image_input = torch.stack(images).to(device)
            image_features = model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            for i in range(logits.shape[0]):
                conf, idx = torch.max(logits[i], dim=0)
                results.append((prompts[idx.item()], conf.item()))
    return results


def clip_classify_labels(
    paths: List[Path],
    prompt_items: List[Dict[str, str]],
    model_name: str = "ViT-B-32",
    pretrained: str = "laion2b_s34b_b79k",
    checkpoint_path: Optional[Path] = None,
    batch_size: Optional[int] = None,
    agg: str = "mean",
    threshold: Optional[float] = None,
) -> List[Tuple[str, float]]:
    model, preprocess, tokenizer, device = load_model(
        model_name, pretrained, checkpoint_path=checkpoint_path
    )
    prompts = [it["prompt"] for it in prompt_items]
    labels = [it["label"] for it in prompt_items]
    cache_prefix = "|".join(
        [
            model_name,
            str(pretrained) if pretrained else "",
            str(checkpoint_path) if checkpoint_path else "",
            device,
        ]
    )
    text_features = _get_text_features(prompts, model, tokenizer, device, cache_prefix)
    label_set = sorted(set(labels))
    label_to_indices: Dict[str, List[int]] = {}
    for i, l in enumerate(labels):
        label_to_indices.setdefault(l, []).append(i)
    results: List[Tuple[str, float]] = []
    with torch.no_grad():
        if not batch_size or batch_size <= 0:
            batch_size = len(paths)
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start : start + batch_size]
            images = []
            for p in batch_paths:
                img = Image.open(p).convert("RGB")
                images.append(preprocess(img))
            image_input = torch.stack(images).to(device)
            image_features = model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            for i in range(logits.shape[0]):
                lp = []
                for l in label_set:
                    idxs = label_to_indices[l]
                    probs = logits[i, idxs]
                    if agg == "max":
                        v = torch.max(probs).item()
                    elif agg == "sum":
                        v = torch.sum(probs).item()
                    else:
                        v = torch.mean(probs).item()
                    lp.append(v)
                li = int(torch.tensor(lp).argmax().item())
                pred_label = label_set[li]
                pred_conf = lp[li]
                if threshold is not None and pred_conf < threshold:
                    results.append(("unknown", pred_conf))
                else:
                    results.append((pred_label, pred_conf))
    return results


def classify_with_probe(
    paths: List[Path],
    probe_path: Path,
    model_name: str = "ViT-B-32",
    pretrained: str = "laion2b_s34b_b79k",
    checkpoint_path: Optional[Path] = None,
    batch_size: Optional[int] = None,
) -> List[Tuple[str, float]]:
    model, preprocess, _, device = load_model(
        model_name, pretrained, checkpoint_path=checkpoint_path
    )
    data = torch.load(str(probe_path), map_location="cpu")
    labels = data["labels"]
    fdim = int(data["feature_dim"]) if "feature_dim" in data else None
    clf = nn.Linear(fdim, len(labels))
    clf.load_state_dict(data["state_dict"])
    clf.eval()
    results: List[Tuple[str, float]] = []
    with torch.no_grad():
        if not batch_size or batch_size <= 0:
            batch_size = len(paths)
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start : start + batch_size]
            images = []
            for p in batch_paths:
                img = Image.open(p).convert("RGB")
                images.append(preprocess(img))
            image_input = torch.stack(images).to(device)
            image_features = model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits = clf(image_features.cpu())
            probs = torch.softmax(logits, dim=1)
            for i in range(probs.shape[0]):
                conf, idx = torch.max(probs[i], dim=0)
                results.append((labels[idx.item()], float(conf.item())))
    return results
