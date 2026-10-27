"""GPU smoke test: load one VLM, ask hand-written questions about two ARKitScenes frames, print the answers.

This checks the path end to end: weights from the local cache, input building, greedy decoding, answer
parsing and the predictions file. It is not an evaluation. The questions are hand-written and have no
ground truth.

Frames are read straight from the upsampling archive of validation video 42444946. Its wide RGB frames
are stored rotated, and the video's sky direction is "Left", so each frame is turned 90 degrees clockwise
before it is fed. That orientation was checked only by eye on a few frames (poster text reads upright, the
floor is at the bottom). It is not verified against the dataset's metadata.

Run one model per process, in a memory-capped scope::

    systemd-run --user --scope -q -p MemoryMax=32G -p MemorySwapMax=2G -- \\
        python scripts/smoke_vlm.py --model qwen3-vl-8b

Predictions go to ``<data root>/outputs/smoke_vlm/<model>_<UTC time>[_blind].jsonl``.
"""

from __future__ import annotations

import argparse
import io
import os
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from measured_space.config import data_root
from measured_space.eval_vlm import (DEFAULT_RESOLUTION, MODELS, VLMRunner, hf_cache_dir, read_predictions,
                                     run_predictions)
from measured_space.qa_schema import QuestionSpec

ARCHIVE = "data/arkitscenes/upsampling/42444946.zip"
SOFA_FRAME = "42444946_223632.654"            # blue two-seat sofa
TABLE_FRAME = "42444946_223677.335"           # round wooden table, a chair behind it, a radiator


def _spec(template: str, qtype: str, frame: str, question: str, answer_type: str,
          choices: tuple[str, ...] | None = None) -> QuestionSpec:
    """A hand-written question about one frame of the smoke-test video.

    Parameters
    ----------
    template : str
        Template id.
    qtype : str
        Question type.
    frame : str
        Frame name (file stem in the archive).
    question : str
        Question text.
    answer_type : str
        ``"number"``, ``"choice"`` or ``"count"``.
    choices : tuple of str or None
        Allowed answers for choice questions.

    Returns
    -------
    QuestionSpec
        The question, with ``image_ref`` pointing into the archive.
    """
    return QuestionSpec(template, qtype, "arkitscenes", "42444946", "wide", frame,
                        f"{ARCHIVE}::42444946/wide/{frame}.png", (), question, answer_type,
                        unit="m" if answer_type == "number" else None, choices=choices,
                        tolerance={"abs": 0.1, "rel": 0.1} if answer_type == "number" else {})


QUESTIONS = [
    _spec("smoke_length_v1", "object_length", SOFA_FRAME,
          "How long is the blue sofa, measured along its longest side, in metres?", "number"),
    _spec("smoke_left_right_v1", "left_right", TABLE_FRAME,
          "In the image, is the wooden chair to the left or to the right of the round table?", "choice",
          ("left", "right")),
    _spec("smoke_count_v1", "count", TABLE_FRAME, "How many chairs are visible in the image?", "count"),
]


def load_frame(spec: QuestionSpec) -> np.ndarray:
    """Read a frame from its archive and turn it upright.

    Parameters
    ----------
    spec : QuestionSpec
        A question whose ``image_ref`` is ``"<zip relative to the data root>::<member>"``.

    Returns
    -------
    numpy.ndarray
        ``(H, W, 3)`` uint8 RGB, rotated 90 degrees clockwise from storage (1440x1920 for these frames).
    """
    archive, member = spec.image_ref.split("::")
    with zipfile.ZipFile(data_root() / archive) as z:
        image = np.asarray(Image.open(io.BytesIO(z.read(member))).convert("RGB"))
    return np.ascontiguousarray(np.rot90(image, k=-1))


def main() -> None:
    """Load the chosen model, ask the smoke questions (and one blind), and print outputs, timing and VRAM.

    With ``--blind-only`` only the blind sofa question is asked, e.g. to check the answer forced after
    cut-off reasoning.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument("--max-new-tokens", type=int, default=None, help="override the model's budget")
    parser.add_argument("--blind-only", action="store_true", help="ask only the blind sofa question")
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_CACHE", str(hf_cache_dir()))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import torch

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = data_root() / "outputs" / "smoke_vlm"
    out_path, blind_path = out_dir / f"{args.model}_{stamp}.jsonl", out_dir / f"{args.model}_{stamp}_blind.jsonl"

    torch.cuda.reset_peak_memory_stats()
    runner = VLMRunner(MODELS[args.model], DEFAULT_RESOLUTION, max_new_tokens=args.max_new_tokens)
    print(f"loaded {runner.info['model_id']} @ {runner.info['snapshot']} in {runner.load_seconds:.1f} s; "
          f"VRAM after load {torch.cuda.memory_allocated() / 2**30:.2f} GiB")
    print("config:", {k: runner.info[k] for k in ("dtype", "attn_implementation", "resolution", "min_pixels",
                                                   "max_pixels", "image_processor", "max_new_tokens",
                                                   "forced_answer_tokens", "reasoning", "transformers",
                                                   "torch")})
    print("orientation: stored frames rotated 90 deg clockwise (sky direction 'Left'); checked by eye only")

    start = time.perf_counter()
    if not args.blind_only:
        run_predictions(runner, QUESTIONS, out_path, load_frame)
    run_predictions(runner, QUESTIONS[:1], blind_path, None, blind=True)
    total = time.perf_counter() - start

    for path in ([] if args.blind_only else [out_path]) + [blind_path]:
        for record in read_predictions(path):
            mode = "blind" if record["blind"] else f"image {record['image_hw']}"
            print(f"\n--- {record['qtype']} ({mode}, {record['n_image_tokens']} image tokens)")
            print(f"raw    : {record['raw_output']!r}")
            if record["forced"]:
                print(f"forced : {record['forced_text']!r}")
            print(f"parsed : {record['parsed']!r} via {record['parse_method']}")
            print(f"latency: {record['latency_s']:.2f} s, {record['n_new_tokens']} new tokens, "
                  f"truncated={record['truncated']}, forced={record['forced']} "
                  f"({record['n_forced_tokens']} forced tokens)")
    print(f"\nall questions: {total:.1f} s; VRAM peak (max_memory_allocated) "
          f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    print(f"predictions: {blind_path if args.blind_only else f'{out_path} and {blind_path}'}")


if __name__ == "__main__":
    main()
