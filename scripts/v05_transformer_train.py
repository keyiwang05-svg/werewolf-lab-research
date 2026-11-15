"""Werewolf Lab v0.5 — supervised Transformer fine-tuning.

Place this file at: scripts/v05_transformer_train.py

Pilot (train/validation only, NEVER reads held-out test samples):
    .venv-transformer/bin/python scripts/v05_transformer_train.py \
      --held-out random --max-train 256 --max-val 64 \
      --epochs 2 --batch-size 4 --run-dir data/v05-transformer/pilot-random-256

Note: this script trains/evaluates RAW PUBLIC TEXT only. Do not claim
comparability with v0.4's masked ablation until that SAME mask operation
is explicitly integrated and both conditions are separately retrained.
"""

import argparse
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as src:
        for chunk in iter(lambda: src.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_partitions(data_dir: Path, held_out: str):
    split_path = data_dir / "split.json"
    dataset_path = data_dir / "dataset.jsonl"
    split = json.loads(split_path.read_text(encoding="utf-8"))

    needed = {"train", "validation", "test"}
    if not needed.issubset(split) or any(not isinstance(split[k], list) for k in needed):
        raise ValueError("split.json must contain train/validation/test seed lists")
    seeds = {name: set(split[name]) for name in needed}
    if any(seeds[a] & seeds[b] for a, b in [
        ("train", "validation"), ("train", "test"), ("validation", "test")
    ]):
        raise ValueError("Seed leakage across split partitions")

    train_rows, val_rows = [], []
    family_names = set()
    all_count = 0
    with dataset_path.open(encoding="utf-8") as src:
        for lineno, line in enumerate(src, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            all_count += 1
            for field in ("game", "seed", "family", "text", "label"):
                if field not in row:
                    raise ValueError(f"Missing {field} in dataset.jsonl line {lineno}")
            if row["label"] not in (0, 1) or not isinstance(row["text"], str):
                raise ValueError(f"Invalid label/text on dataset.jsonl line {lineno}")
            family_names.add(row["family"])
            if row["family"] == held_out:
                continue
            if row["seed"] in seeds["train"]:
                train_rows.append(row)
            elif row["seed"] in seeds["validation"]:
                val_rows.append(row)
            # NEVER load rows assigned to test seed into training or validation.

    if held_out not in family_names:
        raise ValueError(f"Held-out family {held_out!r} not found; options: {sorted(family_names)}")
    if not train_rows or not val_rows:
        raise ValueError("No train/validation rows in requested fold")
    train_games = {(r["family"], r["game"]) for r in train_rows}
    val_games = {(r["family"], r["game"]) for r in val_rows}
    if train_games & val_games:
        raise ValueError("Game leakage between train and validation")
    if {r["seed"] for r in train_rows} & {r["seed"] for r in val_rows}:
        raise ValueError("Seed leakage between train and validation")
    return train_rows, val_rows, {
        "dataset_rows": all_count,
        "families": sorted(family_names),
        "held_out": held_out,
        "train_seed_count": len(seeds["train"]),
        "validation_seed_count": len(seeds["validation"]),
        "test_seed_count": len(seeds["test"]),
        "train_rows_before_limit": len(train_rows),
        "validation_rows_before_limit": len(val_rows),
        "split_sha256": file_sha256(split_path),
        "dataset_sha256": file_sha256(dataset_path),
    }


def choose_subset(rows: list, limit: int, random_seed: int) -> list:
    if limit <= 0 or limit >= len(rows):
        return rows
    rng = random.Random(random_seed)
    indices = sorted(rng.sample(range(len(rows)), limit))
    return [rows[i] for i in indices]


def main():
    parser = argparse.ArgumentParser(description="Train one v0.5 Transformer fold, without peeking at test data")
    parser.add_argument("--held-out", default="random", help="Full strategy family held out of training/validation")
    parser.add_argument("--max-train", type=int, default=0, help="0 = all available training rows")
    parser.add_argument("--max-val", type=int, default=0, help="0 = all available validation rows")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", default="hfl/rbt3")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.max_length < 8 or args.lr <= 0:
        parser.error("epochs, batch-size, lr must be positive; max-length >= 8")
    if args.max_train < 0 or args.max_val < 0:
        parser.error("max-train and max-val must be >= 0")

    # __file__, not shell working directory, resolves the repository's actual location.
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data" / "v04-generalization"
    train_rows, val_rows, meta = load_partitions(data_dir, args.held_out)
    train_rows = choose_subset(train_rows, args.max_train, args.seed)
    val_rows = choose_subset(val_rows, args.max_val, args.seed + 1)
    meta.update({
        "train_rows_used": len(train_rows),
        "validation_rows_used": len(val_rows),
        "train_label_counts": dict(Counter(str(r["label"]) for r in train_rows)),
        "validation_label_counts": dict(Counter(str(r["label"]) for r in val_rows)),
        "train_families_used": sorted({r["family"] for r in train_rows}),
        "validation_families_used": sorted({r["family"] for r in val_rows}),
    })

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = root / run_dir
    # Fail instead of overwriting prior research artifacts.
    if run_dir.exists():
        raise FileExistsError(f"Run output already exists: {run_dir}. Choose a NEW --run-dir.")

    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import transformers

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    print(f"Held-out family: {args.held_out}")
    print(f"Train rows: {len(train_rows)} | Validation rows: {len(val_rows)}")
    print(f"Training families: {meta['train_families_used']}")
    print(f"Device: {device}; epochs={args.epochs}; batch={args.batch_size}")
    print("TEST SET: not accessed, not evaluated")

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.checkpoint, num_labels=2
    ).to(device)

    def collate(batch_rows):
        payload = tokenizer(
            [r["text"] for r in batch_rows],
            padding=True, truncation=True, max_length=args.max_length,
            return_tensors="pt",
        )
        payload["labels"] = torch.tensor([r["label"] for r in batch_rows], dtype=torch.long)
        return {k: value.to(device) for k, value in payload.items()}

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_rows, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate, num_workers=0, generator=generator,
    )
    val_loader = DataLoader(
        val_rows, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate, num_workers=0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Compare a real encoder parameter, not just a task-specific classification head.
    encoder_w = model.bert.encoder.layer[0].attention.self.query.weight
    encoder_start = encoder_w.detach().cpu().clone()

    run_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir = run_dir / "best_model"
    config = vars(args).copy()
    config["device"] = str(device)
    config["pytorch_version"] = torch.__version__
    config["transformers_version"] = transformers.__version__
    config["study_note"] = (
        "Raw public text only; training and validation from non-held-out families. "
        "No test evaluation. Pilot subset is NOT a reportable generalization result."
    )
    (run_dir / "config.json").write_text(
        json.dumps({"config": config, "data": meta}, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    best_val = float("inf")
    metrics = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        seen_train = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            outputs = model(**batch)
            loss = outputs.loss
            if not torch.isfinite(loss).item():
                raise FloatingPointError(f"Non-finite train loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            n = batch["labels"].shape[0]
            train_loss += float(loss.detach().cpu()) * n
            seen_train += n
        train_loss /= seen_train

        model.eval()
        val_loss = 0.0
        seen_val = 0
        with torch.no_grad():
            for batch in val_loader:
                loss = model(**batch).loss
                if not torch.isfinite(loss).item():
                    raise FloatingPointError(f"Non-finite validation loss at epoch {epoch}")
                n = batch["labels"].shape[0]
                val_loss += float(loss.detach().cpu()) * n
                seen_val += n
        val_loss /= seen_val

        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            model.save_pretrained(checkpoint_dir, safe_serialization=True)
            tokenizer.save_pretrained(checkpoint_dir)
        record = {"epoch": epoch, "train_loss": train_loss, "validation_loss": val_loss,
                  "best_so_far": improved}
        metrics.append(record)
        (run_dir / "history.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Epoch {epoch}/{args.epochs} | Train Loss {train_loss:.4f} "
              f"| Val Loss {val_loss:.4f} | {'BEST saved' if improved else 'no change'}", flush=True)

    max_change = (encoder_w.detach().cpu() - encoder_start).abs().max().item()
    print(f"Encoder max abs weight change: {max_change:.9f}")
    print(f"Best validation loss: {best_val:.4f}")
    print(f"Checkpoint saved in: {checkpoint_dir}")
    print("Done. No held-out TEST results have been measured.")
    (run_dir / "training_summary.json").write_text(
        json.dumps({"best_validation_loss": best_val, "final_encoder_max_abs_change": max_change,
                    "epochs_completed": args.epochs, "test_evaluated": False}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
