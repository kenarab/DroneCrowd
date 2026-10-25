"""Strict offline CLIP-EBC construction using pinned upstream inference classes.

No original package __init__ runs: it would download nine backbones. Only the
constructor factories are replaced; the trained forward implementation is kept.
See ops/clip-ebc-sources.json and third_party/clip-ebc-LICENSE.
"""
import gc
import importlib.util
import json
from pathlib import Path
import sys
import types

import torch

from sample_video import sha256


def official_modules(source):
    manifest = json.loads((Path(__file__).resolve().parents[1] / "ops/clip-ebc-sources.json").read_text())
    for name, expected in manifest["files"].items():
        if sha256(source / name) != expected:
            raise ValueError(f"CLIP source hash mismatch: {name}")
    prefix = "dronecrowd_clip_official"
    for suffix, directory in [("", source / "models"), (".clip", source / "models/clip"),
                              (".clip._clip", source / "models/clip/_clip")]:
        module = types.ModuleType(prefix + suffix)
        module.__path__ = [str(directory)]
        sys.modules[module.__name__] = module

    def load(suffix, file):
        spec = importlib.util.spec_from_file_location(prefix + suffix, source / file)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module.__name__] = module
        spec.loader.exec_module(module)
        return module

    load(".utils", "models/utils.py")
    load(".clip.utils", "models/clip/utils.py")
    load(".clip._clip.blocks", "models/clip/_clip/blocks.py")
    visual = load(".clip._clip.image_encoder", "models/clip/_clip/image_encoder.py")
    text = load(".clip._clip.text_encoder", "models/clip/_clip/text_encoder.py")
    tokenizer_module = load(".clip._clip.simple_tokenizer", "models/clip/_clip/simple_tokenizer.py")
    tokenizer = tokenizer_module.SimpleTokenizer()

    def tokenize(prompts):
        result = torch.zeros((len(prompts), 77), dtype=torch.int, device="cpu")
        for index, prompt in enumerate(prompts):
            tokens = [tokenizer.encoder["<|startoftext|>"]] + tokenizer.encode(prompt) + [tokenizer.encoder["<|endoftext|>"]]
            if len(tokens) > 77:
                raise ValueError("Text prompt exceeds 77 tokens")
            result[index, :len(tokens)] = torch.tensor(tokens, device="cpu")
        return result

    factories = sys.modules[prefix + ".clip._clip"]
    factories.tokenize = tokenize
    factories.resnet50_img = lambda **kwargs: visual.ModifiedResNet(
        layers=(3, 4, 6, 3), output_dim=1024, input_resolution=224, width=64,
        heads=32, features_only=True, out_indices=(-1,), reduction=kwargs.get("reduction", 16))
    factories.resnet50_txt = lambda: text.CLIPTextEncoder(1024, 77, 49408, 512, 8, 12)
    factories.vit_b_16_img = lambda **kwargs: visual.VisionTransformer(
        input_resolution=224, patch_size=16, output_dim=512, width=768, layers=12, heads=12, features_only=True)
    factories.vit_b_16_txt = lambda: text.CLIPTextEncoder(512, 77, 49408, 512, 8, 12)
    architecture = load(".clip.model", "models/clip/model.py")
    return architecture, manifest, tokenize


def load_model(path, digest, source, backbone):
    if backbone not in ("resnet50", "vit_b_16"):
        raise ValueError("Only audited CLIP-EBC variants are supported")
    if sha256(path) != digest:
        raise ValueError("CLIP checkpoint hash mismatch")
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(state, dict) or not all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items()):
        raise ValueError("Expected complete CLIP tensor state")
    architecture, manifest, tokenize = official_modules(source)
    config = json.loads((source / "configs/reduction_8.json").read_text())["4"]["nwpu"]
    bins = [tuple(float("inf") if v == "inf" else v for v in pair) for pair in config["bins"]["fine"]]
    anchors = config["anchor_points"]["fine"]["average"]
    # Meta construction avoids allocating a second checkpoint-sized CPU model.
    # Disable the constructor's text cache, which otherwise precedes trained weights.
    with torch.device("meta"):
        model = architecture._clip_ebc(backbone=backbone, bins=bins, anchor_points=anchors,
            reduction=8, freeze_text_encoder=False, prompt_type="word", input_size=224,
            num_vpt=32, deep_vpt=True, vpt_drop=0.)
    expected = model.state_dict()
    if set(state) != set(expected):
        raise ValueError(f"CLIP keys differ: missing={sorted(set(expected)-set(state))[:5]}, extra={sorted(set(state)-set(expected))[:5]}")
    for name, value in state.items():
        if (value.shape != expected[name].shape or value.dtype != expected[name].dtype
                or value.layout != torch.strided or not torch.isfinite(value).all()):
            raise ValueError(f"CLIP tensor incompatible: {name}")
    model.load_state_dict(state, strict=True, assign=True)
    # Restore non-buffer attributes created under the meta context.
    model.anchor_points = torch.tensor(anchors, dtype=torch.float32).view(1, -1, 1, 1)
    model._get_text_prompts()
    model.text_prompts = tokenize(model.text_prompts)
    mask = torch.full((77, 77), -float("inf")).triu_(1)
    for block in model.text_encoder.transformer.resblocks:
        block.attn_mask = mask
    if any(t.is_meta for t in [*model.parameters(), *model.buffers()]):
        raise ValueError("CLIP contains unresolved meta tensors")
    del state, expected
    gc.collect()
    model.eval()
    metadata = {"source_commit": manifest["commit"], "backbone": backbone,
                "checkpoint_sha256": digest, "state_tensor_count": len(model.state_dict()),
                "parameter_count": sum(p.numel() for p in model.parameters()),
                "bins": [[0, 0], [1, 1], [2, 2], [3, 3], [4, "inf"]],
                "anchor_points": anchors, "text_cache": "deferred until after strict checkpoint load"}
    return model, metadata


def prepare_text_cache(model, device):
    model.text_prompts = model.text_prompts.to(device)
    model._extract_text_features()
    model.freeze_text_encoder = True
    if model.text_features.is_meta or not torch.isfinite(model.text_features).all():
        raise ValueError("Invalid trained text cache")
    return sha256_tensor(model.text_features)


def sha256_tensor(tensor):
    import hashlib
    return hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest()
