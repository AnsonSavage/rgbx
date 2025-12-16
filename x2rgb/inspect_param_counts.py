import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

# Avoid permission issues on shared filesystems
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True

import argparse

import torch

# Allow running from any CWD
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from pipeline_x2rgb import StableDiffusionAOVDropoutPipeline  # noqa: E402


MODEL_ID = "zheng95z/x-to-rgb"


@dataclass(frozen=True)
class ParamStats:
    total_params: int
    trainable_params: int
    frozen_params: int


def _count_params(module: torch.nn.Module) -> ParamStats:
    total = 0
    trainable = 0
    for p in module.parameters(recurse=True):
        n = int(p.numel())
        total += n
        if p.requires_grad:
            trainable += n
    return ParamStats(total_params=total, trainable_params=trainable, frozen_params=total - trainable)


def _count_unique_params(modules: Dict[str, Optional[torch.nn.Module]]) -> ParamStats:
    """Count unique parameters across a set of modules (deduplicates shared tensors)."""
    seen: Set[int] = set()
    total = 0
    trainable = 0
    for m in modules.values():
        if m is None:
            continue
        for p in m.parameters(recurse=True):
            pid = id(p)
            if pid in seen:
                continue
            seen.add(pid)
            n = int(p.numel())
            total += n
            if p.requires_grad:
                trainable += n
    return ParamStats(total_params=total, trainable_params=trainable, frozen_params=total - trainable)


def _fmt_int(n: int) -> str:
    return f"{n:,}"


def _fmt_m(n: int) -> str:
    return f"{n / 1e6:.3f}M"


def _print_stats(name: str, stats: ParamStats) -> None:
    print(
        f"{name:<14} total={_fmt_int(stats.total_params):>14} ({_fmt_m(stats.total_params):>10}) | "
        f"trainable={_fmt_int(stats.trainable_params):>14} ({_fmt_m(stats.trainable_params):>10}) | "
        f"frozen={_fmt_int(stats.frozen_params):>14} ({_fmt_m(stats.frozen_params):>10})",
        flush=True,
    )


def _resolve_unet_dir(path: str) -> Path:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = (Path.cwd() / p)
    p = p.absolute()
    if not p.exists():
        raise FileNotFoundError(f"UNet checkpoint path does not exist: {p}")
    if p.is_file():
        raise ValueError(f"UNet checkpoint path must be a directory: {p}")
    # allow passing checkpoint root containing /unet
    if (p / "unet").is_dir():
        return (p / "unet").absolute()
    return p


def main() -> int:
    parser = argparse.ArgumentParser(description="Print parameter counts for x2rgb pipeline components")
    parser.add_argument("--cache_dir", type=str, default=str((_THIS_DIR / "model_cache").absolute()))
    parser.add_argument(
        "--unet_checkpoint",
        type=str,
        default=None,
        help="Optional path to a UNet folder (or checkpoint folder containing /unet) to load instead of base",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32"])
    args = parser.parse_args()

    device = args.device
    dtype = torch.float16 if args.dtype == "float16" else torch.float32

    print(f"Loading pipeline '{MODEL_ID}' on {device} (dtype={args.dtype})", flush=True)
    print(f"cache_dir={Path(args.cache_dir).expanduser().absolute()}", flush=True)

    pipe = StableDiffusionAOVDropoutPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        cache_dir=str(Path(args.cache_dir).expanduser().absolute()),
    ).to(device)

    if args.unet_checkpoint:
        unet_dir = _resolve_unet_dir(args.unet_checkpoint)
        print(f"Loading UNet weights from: {unet_dir}", flush=True)
        unet = pipe.unet
        unet = unet.__class__.from_pretrained(str(unet_dir), torch_dtype=dtype).to(device)
        pipe.unet = unet

    # The pipeline holds modules in these common attributes
    components: Dict[str, Optional[torch.nn.Module]] = {
        "unet": getattr(pipe, "unet", None),
        "vae": getattr(pipe, "vae", None),
        "text_encoder": getattr(pipe, "text_encoder", None),
    }

    print("\nParameter counts:", flush=True)

    total_all = 0
    trainable_all = 0
    frozen_all = 0

    for name, mod in components.items():
        if mod is None:
            print(f"{name:<14} (missing)", flush=True)
            continue
        stats = _count_params(mod)
        _print_stats(name, stats)
        total_all += stats.total_params
        trainable_all += stats.trainable_params
        frozen_all += stats.frozen_params

    # Aggregate unique count (avoids double-counting shared params across modules)
    unique_stats = _count_unique_params(components)
    print("\nAggregate:", flush=True)
    _print_stats("sum(parts)", ParamStats(total_all, trainable_all, frozen_all))
    _print_stats("unique", unique_stats)

    if unique_stats.total_params != total_all:
        print("\nNote: unique != sum(parts) (shared parameters detected).", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
