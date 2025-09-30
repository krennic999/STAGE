import os
import csv
from typing import List
import argparse
from PIL import Image
import numpy as np

def _call_scorer(scorer, images: List[Image.Image], prompts: List[str]):
    out = scorer(images, prompts, None)
    if isinstance(out, dict):
        for k in ("scores", "score", "ocr_scores", "ocr_score"):
            if k in out:
                out = out[k]; break
    return np.asarray(out, dtype=float)

def compute_ocr_scores_to_csv(
    txt_path: str,
    image_root: str,
    scorer,
    output_csv: str,
    batch_size: int = 16,
    exts=(".png", ".jpg", ".jpeg", ".webp")
):
    with open(txt_path, "r", encoding="utf-8") as f:
        prompts = [ln.strip() for ln in f if ln.strip()]

    def find_img(idx: int):
        base = f"{idx:05d}"
        for ext in exts:
            p = os.path.join(image_root, base + ext)
            if os.path.exists(p):
                return p, base + ext
        return None, None

    rows = []
    missing = []
    total, cnt = 0.0, 0

    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start:start + batch_size]
        imgs, valid_prompts, valid_indices, valid_fnames = [], [], [], []

        for j, p in enumerate(batch_prompts):
            idx = start + j
            ipath, fname = find_img(idx)
            if ipath is None:
                missing.append(idx)
                continue
            img = Image.open(ipath).convert("RGB")
            imgs.append(img)
            valid_prompts.append(p)
            valid_indices.append(idx)
            valid_fnames.append(fname)

        if not imgs:
            continue

        scores = _call_scorer(scorer, imgs, valid_prompts)
        total += float(scores.sum())
        cnt   += int(scores.size)

        for k, sc in enumerate(scores.tolist()):
            rows.append((valid_indices[k], valid_fnames[k], valid_prompts[k], sc))

        for im in imgs:
            try: im.close()
            except: pass

    rows.sort(key=lambda r: r[0])
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "filename", "prompt", "ocr_score"])
        w.writerows(rows)

    avg = (total / cnt) if cnt > 0 else float("nan")
    print(f"[OCR] Samples={cnt}, Missing={len(missing)}, AvgScore={avg:.4f}")
    if missing:
        print(f"[OCR] Missing indices (first 20): {missing[:20]}")

    return {"avg": avg, "count": cnt, "missing": missing, "csv": output_csv}


def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ocr_base_root",
        type=str,
        default="/your-ocr-ckpt-root/paddleocr/whl",
        help="PaddleOCR root"
    )
    parser.add_argument(
        "--image_root",
        type=str,
        required=True,
        help="path of generated image root, e.g., 00000.png / 00000.jpg ..."
    )
    parser.add_argument(
        "--txt_path",
        type=str,
        default='/your-code-root/STAGE/data/test_ocr.txt',
        help="txt path of ocr prompts"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="ocr_scores.csv",
        help="csv file to save ocr scores"
    )
    parser.add_argument("--batch_size", type=int, default=16)
    return parser.parse_args()


if __name__ == '__main__':
    from utils.reward_ocr import OcrScorer
    args = build_args()
    scorer = OcrScorer(args)
    try:
        scorer.load_to_device()
    except TypeError:
        scorer.load_to_device('gpu:0')

    res = compute_ocr_scores_to_csv(
        txt_path=args.txt_path,
        image_root=args.image_root,
        scorer=scorer,
        output_csv=args.output_csv,
        batch_size=args.batch_size
    )
    print(res)
