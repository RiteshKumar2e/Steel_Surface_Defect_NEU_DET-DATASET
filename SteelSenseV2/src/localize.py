"""Localization, measured honestly (Reviewer 2, Q1-Q3).

WHAT WENT WRONG IN THE ORIGINAL SUBMISSION
------------------------------------------
Stage E parsed ground-truth boxes out of the XML, rescaled them, replaced only
the label with the classifier's prediction, and scored the result as detection.
That construction has IoU = 1 by definition, which is why AP was identical at
every threshold. It measures rescaling error, not detection. The reviewer is
right, and no version of that number can be reported as detection performance.

WHAT THIS MODULE DOES INSTEAD
-----------------------------
Three quantities are produced and they are kept strictly separate:

  [DET]     A real, class-agnostic proposal detector. Canny at two thresholds
            plus morphological closing yields candidate regions; every proposal
            carries a CONFIDENCE built from measurable region evidence
            (local contrast against its surround, internal edge density, an
            area prior), multiplied by the classifier's class probability for
            the image. Because proposals are ranked by confidence, standard
            all-point-interpolation VOC AP is well defined -- which is the
            direct answer to "a contour detector produces no confidence scores
            with which AP can be computed at all".

  [ORACLE]  Ground-truth boxes carrying the classifier's predicted label. This
            is the CLASSIFICATION CEILING for a perfect localizer, is
            threshold-invariant by construction, and is reported under that
            name only. It is what the original Table 4 actually measured.

  [IOU]     The distribution of IoU between each predicted box and its matched
            ground truth, as a histogram, so the AP numbers can be read against
            the geometry that produced them.

The XML is used ONLY as ground truth. `xml_used_at_inference` is recorded as
False in the output and no code path reads an annotation before predicting.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------


def load_gt_boxes(xml_path: Path, out_size: int) -> List[Dict]:
    """Parse PASCAL-VOC boxes and rescale to the working resolution."""
    if not Path(xml_path).exists():
        return []
    root = ET.parse(str(xml_path)).getroot()
    size = root.find("size")
    W = float(size.findtext("width", "200")) if size is not None else 200.0
    H = float(size.findtext("height", "200")) if size is not None else 200.0
    sx, sy = out_size / W, out_size / H
    out = []
    for obj in root.findall("object"):
        b = obj.find("bndbox")
        if b is None:
            continue
        out.append(
            {
                "label": (obj.findtext("name") or "defect").strip(),
                "box": [
                    float(b.findtext("xmin", "0")) * sx,
                    float(b.findtext("ymin", "0")) * sy,
                    float(b.findtext("xmax", "0")) * sx,
                    float(b.findtext("ymax", "0")) * sy,
                ],
            }
        )
    return out


# ---------------------------------------------------------------------------
# [DET] proposal detector with real confidences
# ---------------------------------------------------------------------------


def propose(gray: np.ndarray, max_props: int = 12) -> List[Dict]:
    """Class-agnostic region proposals, each with a confidence in (0, 1].

    Confidence combines three measurable pieces of evidence, all computed from
    the region itself, so it varies across proposals and can rank them:

        contrast : |mean(inside) - mean(surround)| / std(surround)
        texture  : Canny edge density inside the region
        prior    : a log-normal-ish preference for defect-sized regions

    None of these look at the ground truth.
    """
    h, w = gray.shape
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    cand: List[Tuple[float, List[float]]] = []

    edge_map = cv2.Canny(blur, 50, 150)
    for lo, hi in ((30, 90), (60, 180)):
        e = cv2.Canny(blur, lo, hi)
        for ks in (5, 9):
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (ks, ks))
            closed = cv2.morphologyEx(e, cv2.MORPH_CLOSE, k, iterations=2)
            cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                x, y, bw, bh = cv2.boundingRect(c)
                if bw * bh < 0.002 * h * w:
                    continue
                cand.append((0.0, [float(x), float(y), float(x + bw), float(y + bh)]))

    # Otsu blobs add regions that are contrast- rather than edge-defined.
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        x, y, bw, bh = cv2.boundingRect(c)
        if bw * bh < 0.002 * h * w:
            continue
        cand.append((0.0, [float(x), float(y), float(x + bw), float(y + bh)]))

    scored: List[Dict] = []
    g = gray.astype(np.float32)
    for _, (x0, y0, x1, y1) in cand:
        xi0, yi0 = int(max(0, x0)), int(max(0, y0))
        xi1, yi1 = int(min(w, x1)), int(min(h, y1))
        if xi1 - xi0 < 4 or yi1 - yi0 < 4:
            continue
        inner = g[yi0:yi1, xi0:xi1]
        pad = 10
        sy0, sy1 = max(0, yi0 - pad), min(h, yi1 + pad)
        sx0, sx1 = max(0, xi0 - pad), min(w, xi1 + pad)
        sur = g[sy0:sy1, sx0:sx1]
        contrast = abs(float(inner.mean()) - float(sur.mean())) / (float(sur.std()) + 1e-3)
        texture = float((edge_map[yi0:yi1, xi0:xi1] > 0).mean())
        area = (xi1 - xi0) * (yi1 - yi0) / float(h * w)
        prior = float(np.exp(-((np.log(area + 1e-6) - np.log(0.08)) ** 2) / 2.0))
        conf = (0.5 * np.tanh(contrast) + 0.3 * min(1.0, texture * 5.0) + 0.2 * prior)
        scored.append({"box": [float(xi0), float(yi0), float(xi1), float(yi1)],
                       "score": float(np.clip(conf, 1e-4, 1.0))})

    scored = nms(scored, iou_thr=0.4)[:max_props]
    if not scored:
        # Declared fallback: one whole-image box at low confidence. Counted and
        # reported as `fallback_rate`, never hidden.
        scored = [{"box": [0.0, 0.0, float(w), float(h)], "score": 0.05, "fallback": True}]
    return scored


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return float(inter / (ua + 1e-9))


def nms(dets: List[Dict], iou_thr: float = 0.4) -> List[Dict]:
    keep: List[Dict] = []
    for d in sorted(dets, key=lambda d: -d["score"]):
        if all(iou(d["box"], k["box"]) < iou_thr for k in keep):
            keep.append(d)
    return keep


# ---------------------------------------------------------------------------
# VOC average precision
# ---------------------------------------------------------------------------


def voc_ap(
    preds: List[Dict],
    gts: Dict[str, List[Dict]],
    cls: str,
    iou_thr: float,
) -> Tuple[float, int, int]:
    """All-point-interpolation AP for one class at one IoU threshold.

    preds : [{image_id, label, score, box}], all images, one class filtered here
    gts   : image_id -> [{label, box}]
    """
    p = sorted([d for d in preds if d["label"] == cls], key=lambda d: -d["score"])
    npos = sum(1 for im in gts for g in gts[im] if g["label"] == cls)
    if npos == 0:
        return float("nan"), 0, len(p)
    matched: Dict[str, set] = defaultdict(set)
    tp = np.zeros(len(p))
    fp = np.zeros(len(p))
    for i, d in enumerate(p):
        gt_list = [g for g in gts.get(d["image_id"], []) if g["label"] == cls]
        best, best_j = 0.0, -1
        for j, g in enumerate(gt_list):
            v = iou(d["box"], g["box"])
            if v > best:
                best, best_j = v, j
        if best >= iou_thr and best_j not in matched[d["image_id"]]:
            tp[i] = 1.0
            matched[d["image_id"]].add(best_j)
        else:
            fp[i] = 1.0
    ctp, cfp = np.cumsum(tp), np.cumsum(fp)
    rec = ctp / npos
    prec = ctp / np.maximum(ctp + cfp, 1e-9)
    # all-point interpolation
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.flatnonzero(mrec[1:] != mrec[:-1])
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    return ap, npos, len(p)


def evaluate_detection(
    preds: List[Dict],
    gts: Dict[str, List[Dict]],
    classes: Sequence[str],
    thresholds: Sequence[float] = tuple(np.arange(0.5, 1.0, 0.05)),
) -> Dict:
    per_t: Dict[str, float] = {}
    per_class_50: Dict[str, float] = {}
    for t in thresholds:
        aps = []
        for c in classes:
            ap, npos, _ = voc_ap(preds, gts, c, float(t))
            if not np.isnan(ap):
                aps.append(ap)
                if abs(t - 0.5) < 1e-6:
                    per_class_50[c] = round(100 * ap, 2)
        per_t[f"{t:.2f}"] = float(np.mean(aps) * 100) if aps else float("nan")
    return {
        "AP50": round(per_t["0.50"], 2),
        "AP75": round(per_t["0.75"], 2),
        "mAP@[.5:.95]": round(float(np.nanmean(list(per_t.values()))), 2),
        "per_iou_AP": {k: round(v, 2) for k, v in per_t.items()},
        "per_class_AP50": per_class_50,
        "n_predictions": len(preds),
        "n_gt_boxes": int(sum(len(v) for v in gts.values())),
        "ap_definition": "VOC all-point interpolation, confidence-ranked",
    }


def qualitative_examples(
    preds: List[Dict], gts: Dict[str, List[Dict]], n_good: int = 3, n_bad: int = 3
) -> Dict[str, List[Dict]]:
    """Pick images for a predicted-vs-ground-truth figure, split into good and
    failure cases by IoU (Reviewer 1: Fig. 4 must distinguish predicted boxes
    from ground truth and include several representative failure cases).

    Returns {"good": [...], "bad": [...]}, each entry
    {image_id, pred_box, pred_score, gt_boxes, best_iou}.
    """
    best_per_image: Dict[str, Dict] = {}
    for d in preds:
        cur = best_per_image.get(d["image_id"])
        if cur is None or d["score"] > cur["score"]:
            best_per_image[d["image_id"]] = d

    scored = []
    for im, d in best_per_image.items():
        g = gts.get(im, [])
        best_iou = max([iou(d["box"], x["box"]) for x in g], default=0.0)
        scored.append(
            {
                "image_id": im,
                "pred_box": d["box"],
                "pred_score": d["score"],
                "gt_boxes": [x["box"] for x in g],
                "best_iou": round(best_iou, 4),
            }
        )
    scored.sort(key=lambda r: -r["best_iou"])
    good = scored[:n_good]
    bad = sorted(scored, key=lambda r: r["best_iou"])[:n_bad]
    return {"good": good, "bad": bad}


def iou_histogram(preds: List[Dict], gts: Dict[str, List[Dict]], bins: int = 10) -> Dict:
    """IoU of each image's TOP-scoring prediction against its best-matching GT
    box (Reviewer 2, Q3)."""
    best_per_image: Dict[str, Dict] = {}
    for d in preds:
        cur = best_per_image.get(d["image_id"])
        if cur is None or d["score"] > cur["score"]:
            best_per_image[d["image_id"]] = d
    vals = []
    for im, d in best_per_image.items():
        g = gts.get(im, [])
        vals.append(max([iou(d["box"], x["box"]) for x in g], default=0.0))
    v = np.array(vals) if vals else np.zeros(1)
    hist, edges = np.histogram(v, bins=bins, range=(0.0, 1.0))
    return {
        "n_images": len(vals),
        "mean_iou": round(float(v.mean()), 4),
        "median_iou": round(float(np.median(v)), 4),
        "pct_iou_ge_0.5": round(float((v >= 0.5).mean() * 100), 2),
        "pct_iou_ge_0.75": round(float((v >= 0.75).mean() * 100), 2),
        "histogram_counts": hist.tolist(),
        "histogram_edges": [round(float(e), 2) for e in edges],
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_localization(
    paths: Sequence[str],
    pred_labels: Sequence[str],
    pred_conf: Sequence[float],
    annotation_dir: Path,
    img_size: int,
    classes: Sequence[str],
    max_props: int = 12,
    verbose: bool = True,
) -> Dict:
    """Score [DET] and [ORACLE] on the SAME images, and report both."""
    from features import load_gray

    gts: Dict[str, List[Dict]] = {}
    det_preds: List[Dict] = []
    oracle_preds: List[Dict] = []
    n_fallback = 0
    n_no_gt = 0

    for i, p in enumerate(paths):
        stem = Path(p).stem
        boxes = load_gt_boxes(Path(annotation_dir) / f"{stem}.xml", img_size)
        if not boxes:
            n_no_gt += 1
            continue
        gts[stem] = boxes

        gray = load_gray(p, img_size)
        props = propose(gray, max_props)
        if props and props[0].get("fallback"):
            n_fallback += 1
        for d in props:
            det_preds.append(
                {
                    "image_id": stem,
                    "label": pred_labels[i],
                    "score": float(d["score"] * pred_conf[i]),
                    "box": d["box"],
                }
            )
        for g in boxes:
            oracle_preds.append(
                {"image_id": stem, "label": pred_labels[i],
                 "score": float(pred_conf[i]), "box": list(g["box"])}
            )
        if verbose and (i + 1) % 100 == 0:
            print(f"  localization {i+1}/{len(paths)} images")

    det = evaluate_detection(det_preds, gts, classes)
    orc = evaluate_detection(oracle_preds, gts, classes)
    return {
        "xml_used_at_inference": False,
        "n_images_scored": len(gts),
        "n_images_without_annotation": n_no_gt,
        "proposal_fallback_rate_pct": round(100.0 * n_fallback / max(1, len(gts)), 2),
        "mean_proposals_per_image": round(len(det_preds) / max(1, len(gts)), 2),
        "DET_real_detector": det,
        "DET_iou_histogram": iou_histogram(det_preds, gts),
        "DET_qualitative_examples": qualitative_examples(det_preds, gts),
        "ORACLE_classification_ceiling": {
            **orc,
            "WARNING": (
                "Ground-truth geometry with the predicted label. IoU is 1 by "
                "construction, so this is threshold-invariant and is NOT "
                "detection performance. Report it as a classification ceiling "
                "or not at all."
            ),
        },
    }
