import os
import json
import argparse
import cv2
import torch
import numpy as np

from config import CLASSES, DEFAULT_MODEL_DIR, MODEL_FILE, VOCAB_FILE, LABEL_FILE, MAX_LEN
from features import extract_features, features_to_text, xml_path_for_image, read_xml_boxes, contour_fallback_boxes
from model import SimpleTokenizer, ScratchTinyLLM


def load_model(model_dir):
    tokenizer = SimpleTokenizer.load(os.path.join(model_dir, VOCAB_FILE))
    with open(os.path.join(model_dir, LABEL_FILE), "r", encoding="utf-8") as f:
        labels = json.load(f)
    id_to_label = {int(k): v for k, v in labels["id_to_label"].items()}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ScratchTinyLLM(len(tokenizer.vocab), len(CLASSES)).to(device)
    model.load_state_dict(torch.load(os.path.join(model_dir, MODEL_FILE), map_location=device))
    model.eval()
    return model, tokenizer, id_to_label, device


def predict_image(image_path, model_dir=DEFAULT_MODEL_DIR, dataset_dir=None):
    model, tokenizer, id_to_label, device = load_model(model_dir)
    features = extract_features(image_path)
    text = features_to_text(features)
    ids = torch.tensor([tokenizer.encode(text, MAX_LEN)], dtype=torch.long).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(ids), dim=1)[0].cpu().numpy()
    pred_id = int(np.argmax(probs))
    pred_label = id_to_label[pred_id]
    confidence = float(probs[pred_id])

    xml_path = xml_path_for_image(image_path, dataset_dir)
    boxes = read_xml_boxes(xml_path)
    source = "xml_annotation" if boxes else "contour_fallback"
    if not boxes:
        boxes = contour_fallback_boxes(image_path)

    return {
        "image": image_path,
        "prediction": pred_label,
        "confidence": confidence,
        "feature_text": text,
        "boxes": boxes,
        "box_source": source,
    }


def draw_result(image_path, result, output_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found")
    for item in result["boxes"]:
        x1, y1, x2, y2 = item["box"]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    label = f"{result['prediction']} ({result['confidence']:.2f})"
    cv2.putText(img, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to image")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--dataset", default=None, help="Optional dataset root for XML annotation lookup")
    parser.add_argument("--save", default="results/prediction.jpg")
    args = parser.parse_args()

    result = predict_image(args.image, args.model_dir, args.dataset)
    print(json.dumps(result, indent=2))
    draw_result(args.image, result, args.save)
    print("Saved visual result:", args.save)

if __name__ == "__main__":
    main()
