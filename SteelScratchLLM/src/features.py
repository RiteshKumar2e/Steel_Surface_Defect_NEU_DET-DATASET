import os
import cv2
import math
import numpy as np
import xml.etree.ElementTree as ET
from config import CLASSES, IMG_EXTS

try:
    from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
    SKIMAGE_OK = True
except Exception:
    SKIMAGE_OK = False


def find_image_files(dataset_dir):
    files = []
    for root, _, names in os.walk(dataset_dir):
        for name in names:
            if name.lower().endswith(IMG_EXTS):
                path = os.path.join(root, name)
                label = infer_label_from_path(path)
                if label:
                    files.append((path, label))
    return sorted(files)


def infer_label_from_path(path):
    low = path.replace("\\", "/").lower()
    for c in CLASSES:
        if f"/{c.lower()}/" in low or f"/{c.lower()}_" in low or f"-{c.lower()}" in low:
            return c
    base = os.path.basename(path).lower()
    for c in CLASSES:
        if c.lower() in base:
            return c
    return None


def xml_path_for_image(image_path, dataset_dir=None):
    base = os.path.splitext(os.path.basename(image_path))[0] + ".xml"
    candidates = []
    parent = os.path.dirname(image_path)
    candidates.append(os.path.join(parent, base))
    candidates.append(os.path.join(os.path.dirname(parent), "ANNOTATIONS", base))
    candidates.append(os.path.join(os.path.dirname(os.path.dirname(parent)), "ANNOTATIONS", base))
    if dataset_dir:
        for root, _, names in os.walk(dataset_dir):
            if base in names:
                candidates.append(os.path.join(root, base))
                break
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def read_xml_boxes(xml_path):
    boxes = []
    if not xml_path or not os.path.exists(xml_path):
        return boxes
    try:
        root = ET.parse(xml_path).getroot()
        for obj in root.findall("object"):
            name = obj.findtext("name") or "defect"
            b = obj.find("bndbox")
            if b is None:
                continue
            xmin = int(float(b.findtext("xmin", "0")))
            ymin = int(float(b.findtext("ymin", "0")))
            xmax = int(float(b.findtext("xmax", "0")))
            ymax = int(float(b.findtext("ymax", "0")))
            boxes.append({"label": name, "box": [xmin, ymin, xmax, ymax]})
    except Exception:
        return []
    return boxes


def shannon_entropy(gray):
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    p = hist / (hist.sum() + 1e-9)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def extract_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 60, 160)
    edge_density = float(np.mean(edges > 0))
    entropy = shannon_entropy(gray)
    mean_intensity = float(np.mean(gray))
    std_intensity = float(np.std(gray))

    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas, aspects, boxes = [], [], []
    min_area = max(8, int(0.00015 * h * w))
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        areas.append(float(area))
        aspects.append(float(bw / (bh + 1e-6)))
        boxes.append([int(x), int(y), int(x + bw), int(y + bh)])

    if areas:
        largest_area = max(areas) / (h * w)
        avg_area = float(np.mean(areas)) / (h * w)
        avg_aspect = float(np.mean(aspects))
        max_aspect = float(np.max(aspects))
    else:
        largest_area = avg_area = avg_aspect = max_aspect = 0.0

    glcm_contrast = glcm_homogeneity = 0.0
    lbp_mean = lbp_std = 0.0
    if SKIMAGE_OK:
        small = cv2.resize(gray, (128, 128))
        glcm = graycomatrix(small, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
        glcm_contrast = float(graycoprops(glcm, "contrast")[0, 0])
        glcm_homogeneity = float(graycoprops(glcm, "homogeneity")[0, 0])
        lbp = local_binary_pattern(small, P=8, R=1, method="uniform")
        lbp_mean = float(np.mean(lbp))
        lbp_std = float(np.std(lbp))

    return {
        "width": w,
        "height": h,
        "entropy": entropy,
        "edge_density": edge_density,
        "mean_intensity": mean_intensity,
        "std_intensity": std_intensity,
        "contour_count": len(areas),
        "largest_area": largest_area,
        "avg_area": avg_area,
        "avg_aspect": avg_aspect,
        "max_aspect": max_aspect,
        "glcm_contrast": glcm_contrast,
        "glcm_homogeneity": glcm_homogeneity,
        "lbp_mean": lbp_mean,
        "lbp_std": lbp_std,
        "candidate_boxes": boxes[:10],
    }


def bucket(v, limits):
    for i, t in enumerate(limits):
        if v < t:
            return i
    return len(limits)


def features_to_text(f):
    tokens = []
    tokens.append(f"entropy_{bucket(f['entropy'], [4.5, 5.5, 6.5, 7.2])}")
    tokens.append(f"edge_density_{bucket(f['edge_density'], [0.03, 0.07, 0.12, 0.20])}")
    tokens.append(f"std_intensity_{bucket(f['std_intensity'], [20, 35, 50, 70])}")
    tokens.append(f"contour_count_{bucket(f['contour_count'], [1, 4, 10, 25, 60])}")
    tokens.append(f"largest_area_{bucket(f['largest_area'], [0.002, 0.01, 0.04, 0.12])}")
    tokens.append(f"avg_area_{bucket(f['avg_area'], [0.0005, 0.002, 0.01, 0.05])}")
    tokens.append(f"avg_aspect_{bucket(f['avg_aspect'], [0.7, 1.4, 3.0, 7.0])}")
    tokens.append(f"max_aspect_{bucket(f['max_aspect'], [1.5, 3.0, 6.0, 12.0])}")
    tokens.append(f"glcm_contrast_{bucket(f['glcm_contrast'], [50, 150, 400, 900])}")
    tokens.append(f"homogeneity_{bucket(f['glcm_homogeneity'], [0.1, 0.25, 0.45, 0.7])}")
    tokens.append(f"lbp_std_{bucket(f['lbp_std'], [1.0, 2.0, 3.5, 5.0])}")

    if f["contour_count"] > 25:
        tokens.append("many_small_defects")
    if f["max_aspect"] > 6:
        tokens.append("long_linear_defect")
    if f["largest_area"] > 0.08:
        tokens.append("large_patch_region")
    if f["edge_density"] > 0.12 and f["entropy"] > 6.0:
        tokens.append("web_like_texture")
    return " ".join(tokens)


def contour_fallback_boxes(image_path):
    f = extract_features(image_path)
    return [{"label": "detected_region", "box": b} for b in f["candidate_boxes"]]
