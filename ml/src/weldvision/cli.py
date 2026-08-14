from __future__ import annotations

import argparse
import json
from pathlib import Path

from weldvision.config import load_config
from weldvision.data import (
    exact_duplicate_report,
    group_split,
    read_group_manifest,
)
from weldvision.evaluation import evaluate_checkpoint
from weldvision.export import export_onnx
from weldvision.trainer import run_training


def main() -> None:
    parser = argparse.ArgumentParser(prog="weldvision")
    subparsers = parser.add_subparsers(dest="command", required=True)

    split_parser = subparsers.add_parser("split", help="Create leakage-safe manifests")
    split_parser.add_argument("--manifest", required=True)
    split_parser.add_argument("--output", required=True)
    split_parser.add_argument("--seed", type=int, default=42)

    audit_parser = subparsers.add_parser("audit", help="Audit group and exact duplicate leakage")
    audit_parser.add_argument("--manifest", required=True)
    audit_parser.add_argument("--seed", type=int, default=42)

    train_parser = subparsers.add_parser("train", help="Train source baseline and adaptation")
    train_parser.add_argument("--config", required=True)
    train_parser.add_argument("--device")
    train_parser.add_argument("--resume")

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate a checkpoint")
    evaluate_parser.add_argument("--config", required=True)
    evaluate_parser.add_argument("--checkpoint", required=True)
    evaluate_parser.add_argument("--split", choices=["source_val", "val", "test"], default="test")
    evaluate_parser.add_argument("--device")

    export_parser = subparsers.add_parser("export", help="Export a validated student to ONNX")
    export_parser.add_argument("--config", required=True)
    export_parser.add_argument("--checkpoint", required=True)
    export_parser.add_argument("--output", required=True)

    args = parser.parse_args()
    if args.command == "split":
        _split(args.manifest, args.output, args.seed)
    elif args.command == "audit":
        _audit(args.manifest, args.seed)
    elif args.command == "train":
        path = run_training(
            load_config(args.config),
            device_name=args.device,
            resume=args.resume,
        )
        print(path)
    elif args.command == "evaluate":
        report = evaluate_checkpoint(
            load_config(args.config),
            args.checkpoint,
            split=args.split,
            device_name=args.device,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
    elif args.command == "export":
        path = export_onnx(
            load_config(args.config),
            args.checkpoint,
            args.output,
        )
        print(path)


def _split(manifest_path: str, output_path: str, seed: int) -> None:
    rows = read_group_manifest(manifest_path)
    splits = group_split(rows, seed=seed)
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=True)
    selections = {
        "source_train.txt": [
            row for row in splits["train"] if row.domain == "source" and row.labeled
        ],
        "source_val.txt": [
            row for row in splits["val"] if row.domain == "source" and row.labeled
        ],
        "target_unlabeled.txt": [
            row for row in splits["train"] if row.domain == "target"
        ],
        "target_val.txt": [
            row for row in splits["val"] if row.domain == "target" and row.labeled
        ],
        "target_test.txt": [
            row for row in splits["test"] if row.domain == "target" and row.labeled
        ],
    }
    for name, selected in selections.items():
        (output / name).write_text(
            "\n".join(str(row.image_path) for row in selected) + ("\n" if selected else ""),
            encoding="utf-8",
        )
        print(f"{name}: {len(selected)}")


def _audit(manifest_path: str, seed: int) -> None:
    rows = read_group_manifest(manifest_path)
    splits = group_split(rows, seed=seed)
    duplicates = exact_duplicate_report(splits)
    summary = {
        "groups": len({row.group_id for row in rows}),
        "images": len(rows),
        "splits": {name: len(split_rows) for name, split_rows in splits.items()},
        "cross_split_exact_duplicates": [
            {
                "first": str(first),
                "first_split": first_split,
                "second": str(second),
                "second_split": second_split,
            }
            for first, first_split, second, second_split in duplicates
        ],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if duplicates:
        raise SystemExit("Leakage audit failed")


if __name__ == "__main__":
    main()
