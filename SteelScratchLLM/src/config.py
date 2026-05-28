import os

CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
DEFAULT_DATASET_DIR = "NEU-DET"
DEFAULT_OUTPUT_DIR = "outputs"
DEFAULT_MODEL_DIR = "models"
MODEL_FILE = "scratch_tiny_llm.pt"
VOCAB_FILE = "vocab.json"
LABEL_FILE = "labels.json"
MAX_LEN = 96
SEED = 42
