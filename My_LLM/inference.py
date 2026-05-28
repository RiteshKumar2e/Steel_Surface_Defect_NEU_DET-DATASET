"""
inference.py — generate steel defect descriptions from a trained SteelGPT model.

Usage:
    python inference.py
    python inference.py --prompt "[DEFECT] Type: scratches" --tokens 200 --temp 0.8
"""

import argparse
import os
import torch

import config
from tokenizer import CharTokenizer
from model import SteelGPT

DEFECT_STARTERS = {
    "crazing":         "[DEFECT] Type: crazing.",
    "inclusion":       "[DEFECT] Type: inclusion.",
    "patches":         "[DEFECT] Type: patches.",
    "pitted_surface":  "[DEFECT] Type: pitted_surface.",
    "rolled-in_scale": "[DEFECT] Type: rolled-in_scale.",
    "scratches":       "[DEFECT] Type: scratches.",
}


def load_model(device: torch.device) -> tuple[SteelGPT, CharTokenizer]:
    tokenizer = CharTokenizer()
    if not os.path.exists(config.VOCAB_PATH):
        raise FileNotFoundError(
            f"Vocab not found at {config.VOCAB_PATH}. Run train.py first."
        )
    tokenizer.load(config.VOCAB_PATH)

    best_ckpt = os.path.join(config.CHECKPOINT_DIR, "best_model.pt")
    if not os.path.exists(best_ckpt):
        raise FileNotFoundError(
            f"No checkpoint found at {best_ckpt}. Run train.py first."
        )

    checkpoint = torch.load(best_ckpt, map_location=device)

    model = SteelGPT(
        vocab_size=tokenizer.vocab_size,
        block_size=config.BLOCK_SIZE,
        embed_dim=config.EMBED_DIM,
        num_heads=config.NUM_HEADS,
        num_layers=config.NUM_LAYERS,
        ff_dim=config.FF_DIM,
        dropout=0.0,
    ).to(device)

    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    epoch = checkpoint.get("epoch", "?")
    val_loss = checkpoint.get("val_loss", "?")
    print(f"Loaded checkpoint: epoch={epoch}, val_loss={val_loss}")
    return model, tokenizer


@torch.no_grad()
def generate(model: SteelGPT, tokenizer: CharTokenizer, prompt: str,
             max_new_tokens: int, temperature: float, top_k: int,
             device: torch.device) -> str:
    ids = tokenizer.encode(prompt)
    if not ids:
        raise ValueError("Prompt contains characters not in the vocabulary.")
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out_ids = model.generate(idx, max_new_tokens=max_new_tokens,
                              temperature=temperature, top_k=top_k)
    return tokenizer.decode(out_ids[0].tolist())


def interactive_mode(model: SteelGPT, tokenizer: CharTokenizer, device: torch.device):
    print("\n=== SteelGPT Interactive Mode ===")
    print("Available defect starters:")
    for k in DEFECT_STARTERS:
        print(f"  {k}")
    print("Type a prompt, choose a defect name, or 'quit' to exit.\n")

    while True:
        user_input = input("Prompt> ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if user_input in DEFECT_STARTERS:
            user_input = DEFECT_STARTERS[user_input]

        try:
            result = generate(model, tokenizer, user_input,
                              max_new_tokens=300, temperature=0.8, top_k=40, device=device)
            print("\n--- Generated ---")
            print(result)
            print("-----------------\n")
        except ValueError as e:
            print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="SteelGPT inference")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Seed text (or a defect name like 'crazing')")
    parser.add_argument("--tokens", type=int, default=300,
                        help="Number of new tokens to generate")
    parser.add_argument("--temp", type=float, default=0.8,
                        help="Sampling temperature (lower = more conservative)")
    parser.add_argument("--top_k", type=int, default=40,
                        help="Top-k sampling cutoff")
    parser.add_argument("--interactive", action="store_true",
                        help="Launch interactive REPL mode")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, tokenizer = load_model(device)

    if args.interactive or args.prompt is None:
        interactive_mode(model, tokenizer, device)
    else:
        prompt = DEFECT_STARTERS.get(args.prompt, args.prompt)
        result = generate(model, tokenizer, prompt,
                          max_new_tokens=args.tokens,
                          temperature=args.temp,
                          top_k=args.top_k,
                          device=device)
        print("\n--- Generated ---")
        print(result)
        print("-----------------")


if __name__ == "__main__":
    main()
