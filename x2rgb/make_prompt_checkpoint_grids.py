#!/usr/bin/env python3

"""Create per-scene image grids comparing prompts (X) vs checkpoints (Y).

Expected input directory layout (as in compare_models_* runs):
  <root>/<scene_id>/<prompt_id>/*.json + *.png

Each configuration (scene, prompt, checkpoint) typically has 4 samples (s001..s004).
This script selects exactly ONE image per configuration to keep grids manageable.

Usage:
  python make_prompt_checkpoint_grids.py /path/to/compare_models_08

Outputs:
  Writes one grid PNG per scene into: <root>/grids/

"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


try:
    from PIL import Image, ImageDraw, ImageFont
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Pillow is required. Install it with: pip install pillow (or conda install pillow)"
    ) from exc


_CHECKPOINT_NUM_RE = re.compile(r"(checkpoint-\d+)")
_CHECKPOINT_RUN_RE = re.compile(r"/checkpoints/([^/]+)/")
_SAMPLE_SUFFIX_RE = re.compile(r"__s(\d{3})\.(json|png)$")


@dataclass(frozen=True)
class Item:
    scene_id: str
    prompt_id: str
    prompt: str
    checkpoint_label: str
    sample_key: str  # e.g. s001
    json_path: Path
    image_path: Path


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _sample_key_from_name(path: Path) -> str:
    m = _SAMPLE_SUFFIX_RE.search(path.name)
    if not m:
        return "s000"
    return f"s{m.group(1)}"


def _derive_checkpoint_label(meta: dict) -> str:
    """Derive a stable label from metadata.

    Preference:
      - base -> "base"
      - from checkpoint path: <run_name>/<checkpoint-####>
      - else fallback to checkpoint_id / checkpoint_kind

    run_name is extracted as the path segment right after "/checkpoints/".
    checkpoint number extracted from any "checkpoint-####" component.

    Note: some jsons store checkpoint differently; we consider several fields.
    """

    kind = (meta.get("checkpoint_kind") or "").strip()
    if kind == "base" or (meta.get("checkpoint_id") == "base"):
        return "base"

    candidates: List[str] = []
    for key in ("checkpoint_resolved", "checkpoint", "checkpoint_path"):
        val = meta.get(key)
        if isinstance(val, str) and val.strip():
            candidates.append(val)

    run_name: Optional[str] = None
    ckpt_num: Optional[str] = None

    for p in candidates:
        if run_name is None:
            m = _CHECKPOINT_RUN_RE.search(p)
            if m:
                run_name = m.group(1)
        if ckpt_num is None:
            m2 = _CHECKPOINT_NUM_RE.search(p)
            if m2:
                ckpt_num = m2.group(1)
        if run_name and ckpt_num:
            break

    if run_name and ckpt_num:
        return f"{run_name}/{ckpt_num}"
    if ckpt_num and not run_name:
        return ckpt_num

    ckpt_id = meta.get("checkpoint_id")
    if isinstance(ckpt_id, str) and ckpt_id.strip():
        return ckpt_id

    if kind:
        return kind

    return "unknown"


def _wrap_label(text: str, max_width_px: int, draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont) -> str:
    if not text:
        return ""

    words = text.split()
    lines: List[str] = []
    current: List[str] = []

    def width(s: str) -> int:
        bbox = draw.textbbox((0, 0), s, font=font)
        return bbox[2] - bbox[0]

    for w in words:
        test = (" ".join(current + [w])).strip()
        if current and width(test) > max_width_px:
            lines.append(" ".join(current))
            current = [w]
        else:
            current.append(w)

    if current:
        lines.append(" ".join(current))

    if not lines:
        return text

    return "\n".join(lines)


def _discover_items(root: Path) -> List[Item]:
    items: List[Item] = []

    for scene_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        scene_id = scene_dir.name

        # prompt directories are children of scene
        for prompt_dir in sorted([p for p in scene_dir.iterdir() if p.is_dir()]):
            prompt_id = prompt_dir.name

            for json_path in sorted(prompt_dir.glob("*.json")):
                try:
                    meta = _read_json(json_path)
                except Exception:
                    continue

                prompt = meta.get("prompt") or prompt_id
                if not isinstance(prompt, str):
                    prompt = str(prompt)

                ckpt_label = _derive_checkpoint_label(meta)
                sample_key = _sample_key_from_name(json_path)

                # Prefer local sibling PNG; fall back to meta output_image if present.
                sibling_png = json_path.with_suffix(".png")
                if sibling_png.exists():
                    image_path = sibling_png
                else:
                    out_img = meta.get("output_image")
                    if isinstance(out_img, str) and out_img.strip():
                        image_path = Path(out_img)
                    else:
                        continue

                items.append(
                    Item(
                        scene_id=scene_id,
                        prompt_id=prompt_id,
                        prompt=prompt,
                        checkpoint_label=ckpt_label,
                        sample_key=sample_key,
                        json_path=json_path,
                        image_path=image_path,
                    )
                )

    return items


def _choose_one(items: Sequence[Item], mode: str, index: int, rng: random.Random) -> Optional[Item]:
    if not items:
        return None

    items_sorted = sorted(items, key=lambda it: it.sample_key)

    if mode == "first":
        return items_sorted[0]
    if mode == "index":
        idx0 = max(0, min(len(items_sorted) - 1, index - 1))
        return items_sorted[idx0]
    if mode == "random":
        return rng.choice(items_sorted)

    raise ValueError(f"unknown sample mode: {mode}")


def _load_image(path: Path) -> Image.Image:
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def build_scene_grid(
    scene_id: str,
    scene_items: Sequence[Item],
    out_path: Path,
    sample_mode: str,
    sample_index: int,
    seed: int,
    pad: int,
    header_h: int,
    left_w: int,
    bg: Tuple[int, int, int],
    ) -> None:

    # Group by (prompt, checkpoint_label) -> samples
    by_cell: Dict[Tuple[str, str], List[Item]] = {}
    prompts: Dict[str, str] = {}  # prompt_id -> prompt

    for it in scene_items:
        prompts[it.prompt_id] = it.prompt
        by_cell.setdefault((it.prompt_id, it.checkpoint_label), []).append(it)

    prompt_ids = sorted(prompts.keys(), key=lambda pid: prompts[pid])
    ckpt_labels = sorted({it.checkpoint_label for it in scene_items})

    # Choose one representative per cell.
    chosen: Dict[Tuple[str, str], Item] = {}
    for pid in prompt_ids:
        for ck in ckpt_labels:
            key = (pid, ck)
            rng = random.Random((hash((scene_id, pid, ck, seed)) & 0xFFFFFFFF))
            picked = _choose_one(by_cell.get(key, []), sample_mode, sample_index, rng)
            if picked:
                chosen[key] = picked

    if not chosen:
        return

    # Determine cell image size from first chosen image.
    first_img = _load_image(next(iter(chosen.values())).image_path)
    cell_w, cell_h = first_img.size

    cols = len(prompt_ids)
    rows = len(ckpt_labels)

    grid_w = left_w + pad + cols * cell_w + (cols - 1) * pad + pad
    grid_h = header_h + pad + rows * cell_h + (rows - 1) * pad + pad

    canvas = Image.new("RGB", (grid_w, grid_h), color=bg)
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    # Column headers (prompts)
    for col, pid in enumerate(prompt_ids):
        x0 = left_w + pad + col * (cell_w + pad)
        header_box_w = cell_w
        label = prompts[pid]
        label_wrapped = _wrap_label(label, header_box_w, draw, font)
        draw.multiline_text((x0, 2), label_wrapped, fill=(0, 0, 0), font=font, spacing=2)

    # Row headers (checkpoint labels)
    for row, ck in enumerate(ckpt_labels):
        y0 = header_h + pad + row * (cell_h + pad)
        label_wrapped = _wrap_label(ck, left_w - 6, draw, font)
        # right-align inside left margin
        bbox = draw.multiline_textbbox((0, 0), label_wrapped, font=font, spacing=2)
        text_w = bbox[2] - bbox[0]
        draw.multiline_text((max(2, left_w - 4 - text_w), y0), label_wrapped, fill=(0, 0, 0), font=font, spacing=2)

    # Paste images
    for row, ck in enumerate(ckpt_labels):
        for col, pid in enumerate(prompt_ids):
            key = (pid, ck)
            if key not in chosen:
                continue

            img = _load_image(chosen[key].image_path)
            if img.size != (cell_w, cell_h):
                img = img.resize((cell_w, cell_h), resample=Image.BILINEAR)

            x0 = left_w + pad + col * (cell_w + pad)
            y0 = header_h + pad + row * (cell_h + pad)
            canvas.paste(img, (x0, y0))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main() -> int:
    ap = argparse.ArgumentParser(description="Create per-scene prompt-vs-checkpoint image grids")
    ap.add_argument("root", type=Path, help="Directory like compare_models_08")
    ap.add_argument("--out-dir", type=Path, default=None, help="Output directory (default: <root>/grids)")

    ap.add_argument(
        "--sample",
        choices=["first", "random", "index"],
        default="first",
        help="Choose one of the 4 samples per configuration",
    )
    ap.add_argument("--index", type=int, default=1, help="When --sample=index: choose 1..4 (clamped)")
    ap.add_argument("--seed", type=int, default=0, help="When --sample=random: deterministic seed")

    ap.add_argument("--pad", type=int, default=10, help="Padding between cells")
    ap.add_argument("--header-h", type=int, default=90, help="Top header height for prompt labels")
    ap.add_argument("--left-w", type=int, default=420, help="Left margin width for checkpoint labels")

    args = ap.parse_args()

    root = args.root
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    out_dir = args.out_dir or (root / "grids")

    items = _discover_items(root)
    if not items:
        print(f"No items found under: {root}")
        return 2

    by_scene: Dict[str, List[Item]] = {}
    for it in items:
        by_scene.setdefault(it.scene_id, []).append(it)

    for scene_id, scene_items in sorted(by_scene.items(), key=lambda kv: kv[0]):
        out_path = out_dir / f"grid__{scene_id}.png"
        build_scene_grid(
            scene_id=scene_id,
            scene_items=scene_items,
            out_path=out_path,
            sample_mode=args.sample,
            sample_index=args.index,
            seed=args.seed,
            pad=args.pad,
            header_h=args.header_h,
            left_w=args.left_w,
            bg=(255, 255, 255),
        )
        print(f"Wrote: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
