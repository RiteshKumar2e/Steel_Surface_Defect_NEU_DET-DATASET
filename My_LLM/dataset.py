import os
import xml.etree.ElementTree as ET
import torch
from torch.utils.data import Dataset

import config
from tokenizer import CharTokenizer

DEFECT_DESCRIPTIONS = {
    "crazing": (
        "Crazing is a network of fine cracks spreading across the steel surface. "
        "It typically appears as a web-like pattern and indicates surface fatigue or thermal stress."
    ),
    "inclusion": (
        "Inclusion refers to non-metallic particles embedded within the steel surface. "
        "These foreign materials weaken structural integrity and can lead to crack initiation."
    ),
    "patches": (
        "Patches are irregular regions of discoloration or surface irregularity on the steel. "
        "They may result from uneven rolling pressure or chemical contamination during manufacturing."
    ),
    "pitted_surface": (
        "Pitted surface defects are small cavities or holes on the steel surface. "
        "Pitting is often caused by corrosion, mechanical impact, or gas entrapment during casting."
    ),
    "rolled-in_scale": (
        "Rolled-in scale occurs when oxide scale from the steel surface gets pressed into the material during rolling. "
        "It appears as dark patches or flakes and reduces surface quality significantly."
    ),
    "scratches": (
        "Scratches are linear marks on the steel surface caused by mechanical contact or abrasion. "
        "They vary in depth and length and can compromise the protective coating of the steel."
    ),
}

SEVERITY_MAP = {
    (0, 0.25): "minor",
    (0.25, 0.5): "moderate",
    (0.5, 0.75): "significant",
    (0.75, 1.01): "severe",
}


def _severity(area_ratio: float) -> str:
    for (lo, hi), label in SEVERITY_MAP.items():
        if lo <= area_ratio < hi:
            return label
    return "moderate"


def _region(cx: float, cy: float, w: int, h: int) -> str:
    x_zone = "left" if cx < w / 3 else ("right" if cx > 2 * w / 3 else "center")
    y_zone = "top" if cy < h / 3 else ("bottom" if cy > 2 * h / 3 else "middle")
    return f"{y_zone}-{x_zone}"


def parse_annotation(xml_path: str) -> str:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.findtext("filename", "unknown")
    img_w = int(root.findtext("size/width", "200"))
    img_h = int(root.findtext("size/height", "200"))

    sentences = []
    for obj in root.findall("object"):
        defect = obj.findtext("name", "unknown")
        xmin = int(obj.findtext("bndbox/xmin", "0"))
        ymin = int(obj.findtext("bndbox/ymin", "0"))
        xmax = int(obj.findtext("bndbox/xmax", str(img_w)))
        ymax = int(obj.findtext("bndbox/ymax", str(img_h)))

        bw, bh = xmax - xmin, ymax - ymin
        area_ratio = (bw * bh) / (img_w * img_h)
        cx, cy = xmin + bw / 2, ymin + bh / 2
        region = _region(cx, cy, img_w, img_h)
        severity = _severity(area_ratio)
        desc = DEFECT_DESCRIPTIONS.get(defect, "An unclassified surface defect was detected.")

        sentences.append(
            f"[DEFECT] Type: {defect}. File: {filename}. "
            f"Bounding box: ({xmin},{ymin}) to ({xmax},{ymax}). "
            f"Region: {region}. Severity: {severity}. "
            f"Area coverage: {area_ratio * 100:.1f}%. "
            f"{desc}"
        )

    return "\n".join(sentences)


def build_corpus(annotations_dir: str, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    texts = []
    for fname in sorted(os.listdir(annotations_dir)):
        if fname.endswith(".xml"):
            xml_path = os.path.join(annotations_dir, fname)
            try:
                texts.append(parse_annotation(xml_path))
            except Exception as e:
                print(f"  Skipping {fname}: {e}")

    corpus = "\n\n".join(texts)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(corpus)
    print(f"Corpus saved: {len(corpus):,} chars, {len(texts)} samples -> {output_path}")
    return corpus


class SteelDefectDataset(Dataset):
    def __init__(self, token_ids: list[int], block_size: int, stride: int | None = None):
        self.data = torch.tensor(token_ids, dtype=torch.long)
        self.block_size = block_size
        # stride: distance between successive training windows. Default = block_size//2
        # (50% overlap) — drastically smaller than the 1-char stride used originally.
        self.stride = stride if stride is not None else max(1, block_size // 2)
        self.n_samples = max(1, (len(self.data) - block_size - 1) // self.stride)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        start = idx * self.stride
        x = self.data[start: start + self.block_size]
        y = self.data[start + 1: start + self.block_size + 1]
        return x, y


def load_data(tokenizer: CharTokenizer):
    if not os.path.exists(config.CORPUS_PATH):
        print("Building corpus from XML annotations...")
        corpus = build_corpus(config.ANNOTATIONS_DIR, config.CORPUS_PATH)
    else:
        with open(config.CORPUS_PATH, "r", encoding="utf-8") as f:
            corpus = f.read()
        print(f"Loaded existing corpus: {len(corpus):,} chars")

    if not os.path.exists(config.VOCAB_PATH):
        tokenizer.build_vocab(corpus)
        tokenizer.save(config.VOCAB_PATH)
        print(f"Vocab built: {tokenizer.vocab_size} unique characters")
    else:
        tokenizer.load(config.VOCAB_PATH)
        print(f"Vocab loaded: {tokenizer.vocab_size} unique characters")

    token_ids = tokenizer.encode(corpus)
    split = int(0.9 * len(token_ids))
    train_ds = SteelDefectDataset(token_ids[:split], config.BLOCK_SIZE)
    val_ds = SteelDefectDataset(token_ids[split:], config.BLOCK_SIZE)
    print(f"Train tokens: {split:,} | Val tokens: {len(token_ids) - split:,}")
    return train_ds, val_ds
