import os
import json
import argparse
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm

from config import CLASSES, DEFAULT_DATASET_DIR, DEFAULT_MODEL_DIR, MODEL_FILE, VOCAB_FILE, LABEL_FILE, MAX_LEN, SEED
from features import find_image_files, extract_features, features_to_text
from model import SimpleTokenizer, ScratchTinyLLM

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, label_to_id):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.label_to_id = label_to_id

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        ids = self.tokenizer.encode(self.texts[idx], MAX_LEN)
        y = self.label_to_id[self.labels[idx]]
        return torch.tensor(ids, dtype=torch.long), torch.tensor(y, dtype=torch.long)

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_text_data(dataset_dir):
    items = find_image_files(dataset_dir)
    if not items:
        raise RuntimeError("No labeled images found. Expected class folders like IMAGES/crazing, IMAGES/scratches, etc.")
    texts, labels, paths = [], [], []
    for path, label in tqdm(items, desc="Extracting image features"):
        try:
            f = extract_features(path)
            texts.append(features_to_text(f))
            labels.append(label)
            paths.append(path)
        except Exception as e:
            print(f"Skipping {path}: {e}")
    return texts, labels, paths

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DEFAULT_DATASET_DIR, help="Path to NEU-DET dataset folder")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    set_seed()
    os.makedirs(args.model_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    texts, labels, _ = build_text_data(args.dataset)
    label_to_id = {c: i for i, c in enumerate(CLASSES)}
    id_to_label = {i: c for c, i in label_to_id.items()}

    tokenizer = SimpleTokenizer()
    tokenizer.fit(texts)
    tokenizer.save(os.path.join(args.model_dir, VOCAB_FILE))
    with open(os.path.join(args.model_dir, LABEL_FILE), "w", encoding="utf-8") as f:
        json.dump({"label_to_id": label_to_id, "id_to_label": id_to_label}, f, indent=2)

    stratify = labels if len(set(labels)) > 1 else None
    x_train, x_val, y_train, y_val = train_test_split(texts, labels, test_size=0.2, random_state=SEED, stratify=stratify)
    train_ds = TextDataset(x_train, y_train, tokenizer, label_to_id)
    val_ds = TextDataset(x_val, y_val, tokenizer, label_to_id)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = ScratchTinyLLM(len(tokenizer.vocab), len(CLASSES)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for ids, y in train_loader:
            ids, y = ids.to(device), y.to(device)
            logits = model(ids)
            loss = loss_fn(logits, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()

        model.eval()
        preds, gold = [], []
        with torch.no_grad():
            for ids, y in val_loader:
                ids = ids.to(device)
                logits = model(ids)
                preds.extend(logits.argmax(1).cpu().tolist())
                gold.extend(y.tolist())
        acc = accuracy_score(gold, preds) if gold else 0.0
        print(f"Epoch {epoch:03d} | loss={total_loss/max(1,len(train_loader)):.4f} | val_acc={acc:.4f}")
        if acc >= best_acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(args.model_dir, MODEL_FILE))

    print("\nBest validation accuracy:", best_acc)
    print(classification_report(gold, preds, target_names=CLASSES, zero_division=0))
    print("Saved model to:", os.path.join(args.model_dir, MODEL_FILE))

if __name__ == "__main__":
    main()
