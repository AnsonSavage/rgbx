import os

# Ensure EXR files can be read
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1") # I have no idea what this one is about, but GPT 5.2 added it

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from diffusers import DDIMScheduler
from PIL import Image

sys.dont_write_bytecode = True

# Allow running from any CWD.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from pipeline_x2rgb import StableDiffusionAOVDropoutPipeline  # noqa: E402
from aov_utils import load_aov_image  # noqa: E402


MODEL_ID = "zheng95z/x-to-rgb"
AOV_TYPES_DEFAULT = ["albedo", "normal", "roughness", "metallic", "irradiance"]


def _utc_timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def _stable_hash(text: str, n: int = 8) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]


def _slug(text: str, max_len: int = 80) -> str:
    s = text.strip()
    if not s:
        return "empty"
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if len(s) > max_len:
        s = s[:max_len]
    return s or "empty"


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _normalize_checkpoint_value(value: str) -> str:
    # Allow explicit empty string, "base", or "none" as base model.
    if value is None:
        return ""
    v = value.strip()
    if v.lower() in {"", "base", "none", "original"}:
        return ""
    return value


def _looks_like_diffusers_unet_dir(p: Path) -> bool:
    return (p / "config.json").exists() and (
        (p / "diffusion_pytorch_model.safetensors").exists() or (p / "diffusion_pytorch_model.bin").exists()
    )


def _looks_like_diffusers_pipeline_dir(p: Path) -> bool:
    return (p / "model_index.json").exists()


def _pick_single_safetensors_file(p: Path) -> Optional[Path]:
    if p.is_file() and p.suffix == ".safetensors":
        return p
    if not p.is_dir():
        return None
    candidates = sorted(p.glob("*.safetensors"))
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    preferred = [c for c in candidates if c.name in {"pipeline.safetensors", "model.safetensors", "checkpoint.safetensors"}]
    if len(preferred) == 1:
        return preferred[0]
    raise ValueError(
        f"Checkpoint directory contains multiple .safetensors files; please point to the file explicitly: {p} -> {[c.name for c in candidates]}"
    )


def _resolve_checkpoint(checkpoint_value: str) -> Tuple[str, Optional[Path]]:
    checkpoint_value = _normalize_checkpoint_value(checkpoint_value)
    if checkpoint_value == "":
        return ("base", None)

    p = Path(checkpoint_value).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {p}")

    # Legacy: allow pointing directly to a single safetensors file.
    if p.is_file():
        if p.suffix == ".safetensors":
            return ("pipeline_safetensors", p)
        raise ValueError(f"Checkpoint path must be a directory or a .safetensors file: {p}")

    if not p.is_dir():
        raise ValueError(f"Checkpoint path must be a directory or a .safetensors file: {p}")

    # New convention: if the user points at a folder literally named "unet", treat it as a UNet-only checkpoint.
    if p.name == "unet":
        return ("unet", p)

    # Back-compat: allow passing a checkpoint dir that contains an "unet" folder.
    unet_dir = p / "unet"
    if unet_dir.is_dir() and _looks_like_diffusers_unet_dir(unet_dir):
        return ("unet", unet_dir.resolve())

    # If this looks like a diffusers pipeline directory, load the whole pipeline.
    if _looks_like_diffusers_pipeline_dir(p):
        return ("pipeline_dir", p)

    # Legacy convention: directory containing a single .safetensors file with (some or all) component weights.
    st_path = _pick_single_safetensors_file(p)
    if st_path is not None:
        return ("pipeline_safetensors", st_path)

    # Last resort: treat as UNet dir (diffusers will error with a clearer message).
    return ("unet", p)


def _cast_state_dict(sd: Dict[str, torch.Tensor], dtype: torch.dtype) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if not isinstance(v, torch.Tensor):
            continue
        if v.dtype != dtype:
            out[k] = v.to(dtype=dtype)
        else:
            out[k] = v
    return out


def _load_legacy_safetensors(pipe: StableDiffusionAOVDropoutPipeline, safetensors_path: Path) -> None:
    try:
        from safetensors.torch import load_file  # type: ignore
    except Exception as e:
        raise ImportError(
            "Loading legacy .safetensors checkpoints requires the 'safetensors' package. "
            "Install it in your environment (e.g. `pip install safetensors`)."
        ) from e

    print(f"[pipeline] Loading legacy weights from: {safetensors_path}", flush=True)
    tensors = load_file(str(safetensors_path))

    buckets: Dict[str, Dict[str, torch.Tensor]] = {"unet": {}, "vae": {}, "text_encoder": {}}
    unprefixed: Dict[str, torch.Tensor] = {}
    for k, v in tensors.items():
        if k.startswith("unet."):
            buckets["unet"][k[len("unet."):]] = v
        elif k.startswith("vae."):
            buckets["vae"][k[len("vae."):]] = v
        elif k.startswith("text_encoder."):
            buckets["text_encoder"][k[len("text_encoder."):]] = v
        else:
            unprefixed[k] = v

    loaded_any = False
    if buckets["unet"]:
        result = pipe.unet.load_state_dict(_cast_state_dict(buckets["unet"], pipe.unet.dtype), strict=False)
        print(
            f"[pipeline]   unet: loaded (missing={len(result.missing_keys)} unexpected={len(result.unexpected_keys)})",
            flush=True,
        )
        loaded_any = True
    if buckets["vae"]:
        result = pipe.vae.load_state_dict(_cast_state_dict(buckets["vae"], pipe.vae.dtype), strict=False)
        print(
            f"[pipeline]   vae: loaded (missing={len(result.missing_keys)} unexpected={len(result.unexpected_keys)})",
            flush=True,
        )
        loaded_any = True
    if buckets["text_encoder"]:
        result = pipe.text_encoder.load_state_dict(
            _cast_state_dict(buckets["text_encoder"], pipe.text_encoder.dtype), strict=False
        )
        print(
            f"[pipeline]   text_encoder: loaded (missing={len(result.missing_keys)} unexpected={len(result.unexpected_keys)})",
            flush=True,
        )
        loaded_any = True

    if loaded_any:
        return

    # Fallback: treat it as an unet-only state dict without the "unet." prefix.
    result = pipe.unet.load_state_dict(_cast_state_dict(unprefixed, pipe.unet.dtype), strict=False)
    print(
        f"[pipeline]   unet (unprefixed): loaded (missing={len(result.missing_keys)} unexpected={len(result.unexpected_keys)})",
        flush=True,
    )


def _find_aov_file(aov_dir: Path, aov_type: str, allow_multiple: bool) -> Optional[Path]:
    exts = [".exr", ".png", ".jpg", ".jpeg"]
    matches: List[Path] = []
    for ext in exts:
        matches.extend(sorted(aov_dir.glob(f"*{aov_type}*{ext}")))

    if not matches:
        return None

    if len(matches) == 1 or allow_multiple:
        return matches[0]

    raise ValueError(
        f"Found multiple candidates for AOV '{aov_type}' in {aov_dir}: {[m.name for m in matches]}"
    )


def parse_aov_folder(aov_dir: str, aov_types: List[str], allow_multiple: bool) -> Dict[str, Optional[str]]:
    raw = aov_dir
    # NOTE: Do NOT call .resolve() here: on some systems it can rewrite paths via symlinks/mounts,
    # which makes debugging harder and can point at non-existent canonicalized locations.
    p = Path(aov_dir).expanduser()
    if not p.is_absolute():
        p = (Path.cwd() / p)
    p_abs = p.absolute()
    print(f"[aov]   raw:  {raw}", flush=True)
    print(f"[aov]   abs:  {p_abs}", flush=True)
    if not p_abs.is_dir():
        raise NotADirectoryError(f"AOV folder is not a directory: {p_abs}")

    result: Dict[str, Optional[str]] = {}
    for aov_type in aov_types:
        aov_path = _find_aov_file(p_abs, aov_type, allow_multiple)
        result[aov_type] = str(aov_path) if aov_path is not None else None
    return result


def load_pipeline(
    device: str, cache_dir: str, checkpoint_kind: str, checkpoint_path: Optional[Path]
) -> StableDiffusionAOVDropoutPipeline:
    if checkpoint_kind == "pipeline_dir" and checkpoint_path is not None:
        print(f"[pipeline] Loading full pipeline from: {checkpoint_path}", flush=True)
        pipe = StableDiffusionAOVDropoutPipeline.from_pretrained(
            str(checkpoint_path.resolve()),
            torch_dtype=torch.float16,
        ).to(device)
    else:
        print(
            f"[pipeline] Loading base model '{MODEL_ID}' (device={device}, cache_dir={Path(cache_dir).expanduser().resolve()})",
            flush=True,
        )
        pipe = StableDiffusionAOVDropoutPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            cache_dir=str(Path(cache_dir).expanduser().resolve()),
        ).to(device)

        if checkpoint_kind == "unet" and checkpoint_path is not None:
            unet_checkpoint = checkpoint_path.resolve()
            if not unet_checkpoint.is_dir():
                raise NotADirectoryError(f"UNet checkpoint must be a directory: {unet_checkpoint}")
            print(f"[pipeline] Loading UNet weights from checkpoint: {unet_checkpoint}", flush=True)
            unet = pipe.unet
            unet = unet.__class__.from_pretrained(str(unet_checkpoint), torch_dtype=torch.float16).to(device)
            pipe.unet = unet
        elif checkpoint_kind == "pipeline_safetensors" and checkpoint_path is not None:
            _load_legacy_safetensors(pipe, checkpoint_path.resolve())

    print("[pipeline] Configuring scheduler (DDIM)", flush=True)

    pipe.scheduler = DDIMScheduler.from_config(
        pipe.scheduler.config, rescale_betas_zero_snr=True, timestep_spacing="trailing"
    )
    pipe.set_progress_bar_config(disable=True)
    pipe.to(device)
    return pipe


def _infer_hw_from_any(aovs: Dict[str, Optional[torch.Tensor]], default_hw: Tuple[int, int]) -> Tuple[int, int]:
    for t in aovs.values():
        if t is not None:
            return int(t.shape[1]), int(t.shape[2])
    return default_hw


def _shape_of(t: Optional[torch.Tensor]) -> Optional[List[int]]:
    if t is None:
        return None
    try:
        return [int(x) for x in t.shape]
    except Exception:
        return None


def _checkpoint_id(unet_checkpoint: Optional[Path]) -> str:
    if unet_checkpoint is None:
        return "base"
    s = str(unet_checkpoint)
    # Make IDs nicer when pointing at .../checkpoint-XXXX/unet
    display_name = unet_checkpoint.parent.name if unet_checkpoint.name == "unet" else unet_checkpoint.name
    return f"{_slug(display_name, 60)}__{_stable_hash(s)}"


def _aovset_id(aov_dir: str) -> str:
    p = Path(aov_dir).expanduser()
    if not p.is_absolute():
        p = (Path.cwd() / p)
    p_abs = p.absolute()
    p_str = str(p_abs)
    return f"{_slug(p_abs.name, 60)}__{_stable_hash(p_str)}"


def _prompt_id(prompt: str) -> str:
    return f"{_slug(prompt, 60)}__{_stable_hash(prompt)}"


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def run(args: argparse.Namespace) -> int:
    device = args.device or _default_device()

    # Build lists
    checkpoints = [_normalize_checkpoint_value(x) for x in args.checkpoint]
    aov_folders = args.aov_folder

    prompts: List[str] = []
    if args.prompt:
        prompts.extend(args.prompt)
    if args.prompts_file:
        pf = Path(args.prompts_file).expanduser().resolve()
        if not pf.exists():
            raise FileNotFoundError(f"Prompts file not found: {pf}")
        with pf.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    prompts.append(line)

    if not prompts:
        raise ValueError("Provide at least one prompt via --prompt or --prompts_file")

    run_id = args.run_name.strip() if args.run_name else f"run_{_utc_timestamp()}"
    run_root = Path(args.output_root).expanduser().resolve() / run_id
    _ensure_dir(run_root)

    index_path = run_root / "index.jsonl"

    print("[run] Starting batch_text_only_x2rgb", flush=True)
    print(f"[run] Device: {device}", flush=True)
    print(f"[run] Run root: {run_root}", flush=True)
    print(
        f"[run] Checkpoints: {len(checkpoints)} | AOV folders: {len(aov_folders)} | Prompts: {len(prompts)}",
        flush=True,
    )
    print(
        f"[run] Steps={args.inference_steps} seed={args.seed} guidance_scale={args.guidance_scale} image_guidance_scale={args.image_guidance_scale}",
        flush=True,
    )
    print(f"[run] Samples per configuration: {int(args.num_samples)}", flush=True)

    # Resolve checkpoints once
    resolved_checkpoints: List[Tuple[str, str, Optional[Path]]] = []
    for cp in checkpoints:
        print(f"[run] Resolving checkpoint: {cp!r}", flush=True)
        kind, path = _resolve_checkpoint(cp)
        resolved_checkpoints.append((cp, kind, path))

    # Iterate checkpoint-major to minimize model reload overhead
    total_checkpoints = len(resolved_checkpoints)
    total_aov_folders = len(aov_folders)
    total_prompts = len(prompts)

    for cp_index, (cp_raw, cp_kind, cp_path) in enumerate(resolved_checkpoints, start=1):
        cp_id = _checkpoint_id(cp_path)
        print(
            f"\n=== Checkpoint {cp_index}/{total_checkpoints}: {cp_id} ({cp_kind}: {'base' if cp_path is None else cp_path}) ===",
            flush=True,
        )

        pipe = load_pipeline(device=device, cache_dir=args.cache_dir, checkpoint_kind=cp_kind, checkpoint_path=cp_path)

        for aov_index, aov_folder in enumerate(aov_folders, start=1):
            print(f"[aov] ({aov_index}/{total_aov_folders}) Parsing AOV folder: {aov_folder}", flush=True)
            aovset = parse_aov_folder(aov_folder, args.aov_types, allow_multiple=args.allow_multiple_aovs)

            missing = [k for k, v in aovset.items() if v is None]
            if missing:
                print(f"[aov] Missing AOVs in folder (will be None): {missing}", flush=True)
            for k, v in aovset.items():
                if v is not None:
                    print(f"[aov]   {k}: {v}", flush=True)

            # Load AOV tensors
            aov_tensors: Dict[str, Optional[torch.Tensor]] = {}
            for aov_type in args.aov_types:
                print(f"[aov] Loading {aov_type}...", flush=True)
                aov_tensors[aov_type] = load_aov_image(aovset[aov_type], aov_type, device)
                t = aov_tensors[aov_type]
                if t is None:
                    print(f"[aov]   {aov_type}: None", flush=True)
                else:
                    print(f"[aov]   {aov_type}: shape={tuple(int(x) for x in t.shape)}", flush=True)

            height, width = _infer_hw_from_any(aov_tensors, default_hw=(args.default_height, args.default_width))
            print(f"[aov] Using resolution: {width}x{height}", flush=True)

            # Prepare output folder (compare across checkpoints inside same prompt folder)
            aov_id = _aovset_id(aov_folder)

            required_aovs = list(args.aov_types)

            for prompt_index, prompt in enumerate(prompts, start=1):
                prompt_dir = run_root / aov_id / _prompt_id(prompt)
                _ensure_dir(prompt_dir)

                num_samples = int(args.num_samples)
                if num_samples < 1:
                    raise ValueError("--num_samples must be >= 1")

                generators: List[torch.Generator] = []
                seed_values: List[int] = []
                if int(args.seed) == -1:
                    for _ in range(num_samples):
                        g = torch.Generator(device=device)
                        s = int(g.seed())
                        g.manual_seed(s)
                        generators.append(g)
                        seed_values.append(s)
                else:
                    base_seed = int(args.seed)
                    for i in range(num_samples):
                        g = torch.Generator(device=device)
                        s = base_seed + i
                        g.manual_seed(s)
                        generators.append(g)
                        seed_values.append(s)

                seed_preview = seed_values[0] if seed_values else int(args.seed)
                print(
                    f"[infer] ({prompt_index}/{total_prompts}) checkpoint={cp_id} aovset={aov_id} samples={num_samples} seed0={seed_preview} prompt={prompt!r}",
                    flush=True,
                )

                result = pipe(
                    prompt=prompt,
                    albedo=aov_tensors.get("albedo"),
                    normal=aov_tensors.get("normal"),
                    roughness=aov_tensors.get("roughness"),
                    metallic=aov_tensors.get("metallic"),
                    irradiance=aov_tensors.get("irradiance"),
                    num_inference_steps=args.inference_steps,
                    height=height,
                    width=width,
                    num_images_per_prompt=num_samples,
                    generator=generators,
                    required_aovs=required_aovs,
                    guidance_scale=args.guidance_scale,
                    image_guidance_scale=args.image_guidance_scale,
                    guidance_rescale=0.7,
                    output_type="np",
                )

                print("[infer] Completed diffusion; saving outputs...", flush=True)

                images = result.images
                if not isinstance(images, np.ndarray):
                    images = np.asarray(images)

                for sample_index in range(num_samples):
                    sample_tag = "" if num_samples == 1 else f"__s{sample_index + 1:03d}"
                    out_path = prompt_dir / f"{cp_id}{sample_tag}.png"
                    meta_path = prompt_dir / f"{cp_id}{sample_tag}.json"

                    img_np = images[sample_index]
                    img_u8 = (np.clip(img_np, 0.0, 1.0) * 255).astype(np.uint8)
                    Image.fromarray(img_u8).save(out_path)

                    metadata = {
                        "created_at": datetime.utcnow().isoformat() + "Z",
                        "run_id": run_id,
                        "model_id": MODEL_ID,
                        "checkpoint": cp_raw,
                        "checkpoint_resolved": None if cp_path is None else str(cp_path),
                        "checkpoint_kind": cp_kind,
                        "checkpoint_id": cp_id,
                        "aov_folder": str(Path(aov_folder).expanduser().absolute()),
                        "aovset_id": aov_id,
                        "prompt": prompt,
                        "prompt_id": _prompt_id(prompt),
                        "seed": int(seed_values[sample_index]) if seed_values else int(args.seed),
                        "sample_index": int(sample_index),
                        "num_samples": int(num_samples),
                        "device": device,
                        "inference_steps": int(args.inference_steps),
                        "guidance_scale": float(args.guidance_scale),
                        "image_guidance_scale": float(args.image_guidance_scale),
                        "cache_dir": str(Path(args.cache_dir).expanduser().resolve()),
                        "height": int(height),
                        "width": int(width),
                        "aovs": {
                            aov_type: {
                                "path": aovset[aov_type],
                                "shape": _shape_of(aov_tensors[aov_type]),
                            }
                            for aov_type in args.aov_types
                        },
                        "output_image": str(out_path),
                    }

                    with meta_path.open("w", encoding="utf-8") as f:
                        json.dump(metadata, f, indent=2)

                    with index_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(metadata) + "\n")

                    print(f"[save] Wrote: {out_path}", flush=True)
                    print(f"[save] Meta:  {meta_path}", flush=True)

    print("\n[run] Done.", flush=True)
    return 0


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Batch runner for x2rgb: loops over checkpoints × AOV folders × prompts and saves outputs + metadata."
    )

    p.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        help=(
            "Checkpoint path. Supports: (1) UNet-only checkpoint dir (e.g. .../checkpoint-2300/unet), "
            "(2) checkpoint dir containing an 'unet' subdir, or (3) legacy full checkpoint as a single .safetensors file "
            "(either pass the file directly, or pass a directory containing exactly one .safetensors). "
            "Repeat to provide multiple. Use empty string (\"\"), 'base', or 'none' for the original model."
        ),
    )
    p.add_argument(
        "--aov_folder",
        action="append",
        default=[],
        help="Folder containing AOV images (expects files like albedo*.png/.exr, normal*.png/.exr, etc.). Repeatable.",
    )

    p.add_argument(
        "--prompt",
        action="append",
        default=[],
        help="Prompt string. Repeatable.",
    )
    p.add_argument(
        "--prompts_file",
        type=str,
        default=None,
        help="Text file with one prompt per line (optional; added to any --prompt entries).",
    )

    p.add_argument("--output_root", type=str, default="./outputs/batch_text_only_x2rgb", help="Root output directory")
    p.add_argument("--run_name", type=str, default=None, help="Optional run name (otherwise timestamped)")

    p.add_argument("--device", type=str, default=None, help="cuda or cpu (default: auto)")
    p.add_argument("--cache_dir", type=str, default="./model_cache", help="HuggingFace cache dir")

    p.add_argument("--seed", type=int, default=-1, help="Seed used for every generation (use -1 for random)")
    p.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of images to sample per checkpoint × AOV folder × prompt (default: 1).",
    )
    p.add_argument("--inference_steps", type=int, default=100)
    p.add_argument("--guidance_scale", type=float, default=7.5)
    p.add_argument("--image_guidance_scale", type=float, default=1.5)

    p.add_argument(
        "--aov_types",
        nargs="+",
        default=AOV_TYPES_DEFAULT,
        help=f"AOV types to look for (default: {AOV_TYPES_DEFAULT})",
    )
    p.add_argument(
        "--allow_multiple_aovs",
        action="store_true",
        help="If multiple files match an AOV type in a folder, pick the first (sorted) instead of erroring.",
    )

    p.add_argument("--default_height", type=int, default=768)
    p.add_argument("--default_width", type=int, default=768)

    return p


if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()

    if not args.checkpoint:
        # Default to base if user provides none
        args.checkpoint = [""]
    if not args.aov_folder:
        raise SystemExit("Provide at least one --aov_folder")

    raise SystemExit(run(args))
