import os
import argparse
import json
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from config import DEFAULT_DATASET_DIR
from features import find_image_files
from predict import predict_image, draw_result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--out", default="results")
    parser.add_argument("--draw", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    items = find_image_files(args.dataset)
    rows, y_true, y_pred = [], [], []
    for path, true_label in tqdm(items, desc="Evaluating"):
        try:
            res = predict_image(path, args.model_dir, args.dataset)
            rows.append({
                "image": path,
                "true_label": true_label,
                "predicted_label": res["prediction"],
                "confidence": res["confidence"],
                "box_source": res["box_source"],
                "feature_text": res["feature_text"],
            })
            y_true.append(true_label)
            y_pred.append(res["prediction"])
            if args.draw:
                name = os.path.basename(path)
                draw_result(path, res, os.path.join(args.out, name))
        except Exception as e:
            print("Error:", path, e)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.out, "evaluation_results.csv")
    df.to_csv(csv_path, index=False)
    report = classification_report(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred) if y_true else 0.0
    with open(os.path.join(args.out, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {acc}\n\n")
        f.write(report)
    print("Accuracy:", acc)
    print(report)
    print("Saved:", csv_path)

if __name__ == "__main__":
    main()
