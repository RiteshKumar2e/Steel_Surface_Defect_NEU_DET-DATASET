import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ANNOTATIONS_DIR = os.path.join(BASE_DIR, "..", "NEU-DET", "ANNOTATIONS")
CORPUS_PATH = os.path.join(BASE_DIR, "steel_corpus.txt")
VOCAB_PATH = os.path.join(BASE_DIR, "vocab.json")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

# Transformer hyperparameters
BLOCK_SIZE = 128       # context window (characters)
EMBED_DIM = 128        # embedding dimension
NUM_HEADS = 4          # attention heads
NUM_LAYERS = 4         # transformer blocks
FF_DIM = 512           # feed-forward hidden dim
DROPOUT = 0.1

# Training
BATCH_SIZE = 64
LEARNING_RATE = 3e-4
MAX_EPOCHS = 5         # reduced for CPU; corpus is small + highly repetitive
EVAL_INTERVAL = 1      # evaluate every epoch so best_model.pt saves quickly
CHECKPOINT_EVERY = 5   # save checkpoint every N epochs

DEVICE = "cuda"        # falls back to cpu in train.py if unavailable
