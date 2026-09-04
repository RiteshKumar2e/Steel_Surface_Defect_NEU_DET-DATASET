"""Dataset registry and experiment configuration for SteelSense-BiLSTM v2.

Everything a reviewer needs to reproduce a number is declared here: the class
lists, the split ratios, the seeds and the feature/binning versions. Nothing is
hard-coded further down the pipeline.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
PKG = Path(__file__).resolve().parents[1]
CACHE_DIR = PKG / "cache"
RESULTS_DIR = PKG / "results"

# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

NEU_CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]

# The six SteelDefectX folders whose names mirror NEU-DET. The duplicate audit
# (src/duplicates.py) shows four of them are pixel copies of NEU-DET, so this
# subset exists ONLY to reproduce the original submission's protocol.
SDX_PAPER_CLASSES = [
    "Crazing",
    "Inclusion",
    "Patches",
    "Pitted surface",
    "Rolled in scale",
    "Scratches",
]

SDX_ALL_CLASSES = [
    "Bright scratch",
    "Crazing",
    "Crease",
    "Crescent gap",
    "Finishing roll printing",
    "Inclusion",
    "Iron scale compression",
    "Iron sheet ash",
    "Oil spot",
    "Oxide scale of plate system",
    "Oxide scale of temperature system",
    "Patches",
    "Pitted surface",
    "Punching",
    "Red iron sheet",
    "Rolled in scale",
    "Rolled pit",
    "Scratches",
    "Secondary rust skin",
    "Silk spot",
    "Slag inclusion",
    "Waist folding",
    "Welding line",
    "White rust",
]


@dataclass
class DatasetSpec:
    name: str
    image_root: Path
    classes: List[str]
    img_size: int
    annotation_dir: Optional[Path] = None
    # Drop any image that duplicates an image in this other dataset.
    drop_duplicates_of: Optional[str] = None
    # Minimum images a class must keep after de-duplication to stay in the task.
    min_class_size: int = 20
    note: str = ""


DATASETS: Dict[str, DatasetSpec] = {
    "neu_det": DatasetSpec(
        name="neu_det",
        image_root=ROOT / "NEU-DET" / "IMAGES",
        classes=NEU_CLASSES,
        img_size=200,
        annotation_dir=ROOT / "NEU-DET" / "ANNOTATIONS",
        note="1800 images, 300/class, PASCAL-VOC boxes for every image.",
    ),
    # Appendix option: the full 24-class SteelDefectX with every image that
    # duplicates NEU-DET removed (3545 images, 19 classes at usable size).
    # Swap it into notebook 02 with a one-line config change if a genuinely
    # NEU-disjoint evaluation is wanted.
    "steeldefectx": DatasetSpec(
        name="steeldefectx",
        image_root=ROOT / "SteelDefectX" / "train_by_class",
        classes=SDX_ALL_CLASSES,
        img_size=256,
        drop_duplicates_of="neu_det",
        note="24 classes, NEU-DET duplicates removed; 19 retain >= 20 images.",
    ),
    # The six-class subset used in the manuscript. This is the SteelDefectX
    # configuration notebook 02 reports. Its train/test split is built the same
    # way as every other -- image level, duplicate-grouped -- so results on it
    # are internally valid. What the NEU-DET overlap rules out is calling it an
    # INDEPENDENT second corpus, or claiming cross-dataset transfer; see
    # notebook 02 section 9b, which measures the overlap rather than ignoring it.
    "steeldefectx_paper": DatasetSpec(
        name="steeldefectx_paper",
        image_root=ROOT / "SteelDefectX" / "train_by_class",
        classes=SDX_PAPER_CLASSES,
        img_size=256,
        drop_duplicates_of=None,
        note=("6-class subset as used in the manuscript, 1631 images. "
              "64.1% of it also appears in NEU-DET -- report it as a related "
              "benchmark, never as independent evidence."),
    ),
    "steeldefectx_paper_clean": DatasetSpec(
        name="steeldefectx_paper_clean",
        image_root=ROOT / "SteelDefectX" / "train_by_class",
        classes=SDX_PAPER_CLASSES,
        img_size=256,
        drop_duplicates_of="neu_det",
        note=("The 6-class subset after NEU-DET duplicate removal. Only two "
              "classes survive at usable size, which is itself the finding; "
              "used in notebook 02 section 9b to size the overlap effect."),
    ),
}

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------


@dataclass
class SplitConfig:
    """Image-level, stratified, three-way. Applied BEFORE augmentation and
    BEFORE feature extraction -- see src/splits.py."""

    train: float = 0.70
    val: float = 0.10
    test: float = 0.20
    seed: int = 20240501
    # Near-duplicate handling: duplicate groups are assigned to a single split
    # so no image can appear on both sides of the train/test boundary.
    dedupe: str = "group"  # {"group", "drop", "report"}
    dhash_bits: int = 8
    # Intra-dataset thresholds. Calibrated on NEU-DET: at (2, 0.98) the union
    # of near-duplicate pairs produces no cross-class group, i.e. no pair that
    # a human would call two different defects is ever merged. Looser settings
    # chain low-texture images together transitively.
    dup_hamming_max: int = 2
    dup_corr_min: float = 0.98
    # Cross-dataset thresholds are looser on the hash because the SteelDefectX
    # copies were re-encoded at a different resolution, and tighter on the
    # correlation because a false positive there deletes a legitimate image.
    xdup_hamming_max: int = 6
    xdup_corr_min: float = 0.97
    # Never merge two images of different classes into one duplicate group;
    # report them instead (they are label-consistency findings, not duplicates).
    group_same_class_only: bool = True
    # NEU-DET only: reuse the frozen split the CNN baselines already used.
    reuse_frozen_split: Optional[Path] = ROOT / "paper_results" / "splits" / "split_v1.csv"


@dataclass
class FeatureConfig:
    version: str = "v2"
    img_size: int = 200
    glcm_levels: int = 32
    glcm_distances: tuple = (1, 3)
    lbp_points: int = 8
    lbp_radius: int = 1


@dataclass
class BinConfig:
    """Discretization is fitted on TRAIN ONLY (see src/discretize.py)."""

    n_bins: int = 5
    strategy: str = "quantile"  # {"quantile", "uniform"}
    # Values outside the fitted range at inference are clamped into the end
    # bins and counted; the counts are written to the run record.
    oor_policy: str = "clamp"  # {"clamp", "oor_token"}


@dataclass
class ModelConfig:
    arch: str = "bilstm"  # {"bilstm", "deepsets", "bilstm_numeric"}
    embed_dim: int = 96
    hidden_dim: int = 192
    num_layers: int = 2
    dropout: float = 0.30
    max_len: int = 160
    # Which pooled view(s) feed the head: "all" (attention+max+mean, deployed)
    # or a single one of "attention" / "max" / "mean" -- the ablation switch
    # for Reviewer 1 Q1 / Reviewer 2 Q6.
    pooling: str = "all"


@dataclass
class TrainConfig:
    epochs: int = 40
    batch_size: int = 32
    lr: float = 8e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.06
    grad_clip: float = 1.0
    pct_start: float = 0.15
    # Image-space augmentation, TRAIN SPLIT ONLY.
    aug_variants: int = 3
    # Checkpoint selection uses validation macro-F1 only. The test split is
    # loaded once, after selection is frozen.
    select_metric: str = "val_macro_f1"
    ensemble_size: int = 5
    early_stop_patience: int = 15
    class_weighting: bool = True


@dataclass
class ExperimentConfig:
    dataset: str = "neu_det"
    seeds: List[int] = field(default_factory=lambda: [42, 1337, 2024, 7, 20250101])
    split: SplitConfig = field(default_factory=SplitConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    bins: BinConfig = field(default_factory=BinConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    tag: str = "main"

    def spec(self) -> DatasetSpec:
        return DATASETS[self.dataset]

    def out_dir(self) -> Path:
        d = RESULTS_DIR / self.dataset / self.tag
        d.mkdir(parents=True, exist_ok=True)
        return d

    def to_dict(self) -> dict:
        d = asdict(self)
        d["split"]["reuse_frozen_split"] = (
            str(self.split.reuse_frozen_split) if self.split.reuse_frozen_split else None
        )
        return d


def torch_threads() -> int:
    return int(os.environ.get("SS_THREADS", os.cpu_count() or 4))
