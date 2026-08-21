from __future__ import annotations

import argparse
import json
from pathlib import Path

from weldvision.config import generate_b0_factor_ablations, load_config
from weldvision.data import (
    exact_duplicate_report,
    group_split,
    read_group_manifest,
)
from weldvision.evaluation import evaluate_checkpoint
from weldvision.export import export_onnx
from weldvision.lohi import prepare_lohi
from weldvision.quality_training import train_quality_gate
from weldvision.rgb_challenge import prepare_rgb_challenge
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

    lohi_parser = subparsers.add_parser(
        "prepare-lohi",
        help="Extract, validate, and create LoHi-WELD manifests",
    )
    lohi_parser.add_argument("--archive", required=True)
    lohi_parser.add_argument("--destination", required=True)
    lohi_parser.add_argument("--manifests", required=True)
    lohi_parser.add_argument("--fold", type=int, default=0)
    lohi_parser.add_argument("--seed", type=int, default=42)

    rgb_parser = subparsers.add_parser(
        "prepare-rgb-challenge",
        help="Prepare the small external RGB weld challenge set",
    )
    rgb_parser.add_argument("--archive", required=True)
    rgb_parser.add_argument("--destination", required=True)
    rgb_parser.add_argument("--manifests", required=True)

    train_parser = subparsers.add_parser("train", help="Train source baseline and adaptation")
    train_parser.add_argument("--config", required=True)
    train_parser.add_argument("--device")
    train_parser.add_argument("--resume")

    quality_parser = subparsers.add_parser(
        "train-quality",
        help="Train the smartphone capture quality gate",
    )
    quality_parser.add_argument("--manifest", required=True)
    quality_parser.add_argument("--output", required=True)
    quality_parser.add_argument("--epochs", type=int, default=20)
    quality_parser.add_argument("--batch-size", type=int, default=16)
    quality_parser.add_argument("--device")

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate a checkpoint")
    evaluate_parser.add_argument("--config", required=True)
    evaluate_parser.add_argument("--checkpoint", required=True)
    evaluate_parser.add_argument(
        "--split",
        choices=["source_val", "source_test", "val", "test"],
        default="test",
    )
    evaluate_parser.add_argument("--device")

    export_parser = subparsers.add_parser("export", help="Export a validated student to ONNX")
    export_parser.add_argument("--config", required=True)
    export_parser.add_argument("--checkpoint", required=True)
    export_parser.add_argument("--output", required=True)

    train_yolo_parser = subparsers.add_parser(
        "train-yolo",
        help="Train research-only Ultralytics YOLO on the frozen B0 split",
    )
    train_yolo_parser.add_argument("--config", required=True)
    train_yolo_parser.add_argument("--device")
    train_yolo_parser.add_argument("--resume")

    evaluate_yolo_parser = subparsers.add_parser(
        "evaluate-yolo",
        help="Evaluate a YOLO checkpoint with the SSDLite B0 metrics",
    )
    evaluate_yolo_parser.add_argument("--config", required=True)
    evaluate_yolo_parser.add_argument("--checkpoint", required=True)
    evaluate_yolo_parser.add_argument(
        "--split",
        choices=["source_val", "source_test", "val", "test"],
        default="source_test",
    )
    evaluate_yolo_parser.add_argument("--device")

    ablation_parser = subparsers.add_parser(
        "generate-ablations",
        help="Generate one-factor B0 ablation configs",
    )
    ablation_parser.add_argument("--base", required=True)
    ablation_parser.add_argument("--output", required=True)
    ablation_parser.add_argument("--results-root", required=True)

    explain_parser = subparsers.add_parser(
        "explain-network",
        help="Write an HTML walkthrough of each YOLOv8s inference step",
    )
    explain_parser.add_argument("--output", required=True)
    explain_parser.add_argument(
        "--image",
        help="Optional weld photo. If omitted, a synthetic bead is generated.",
    )
    explain_parser.add_argument(
        "--checkpoint",
        help="Optional YOLOv8s weights (student_best.pt) for live activation maps",
    )
    explain_parser.add_argument("--device")
    explain_parser.add_argument("--image-size", type=int, default=640)
    explain_parser.add_argument("--confidence", type=float, default=0.25)

    args = parser.parse_args()
    if args.command == "split":
        _split(args.manifest, args.output, args.seed)
    elif args.command == "audit":
        _audit(args.manifest, args.seed)
    elif args.command == "prepare-lohi":
        report = prepare_lohi(
            args.archive,
            args.destination,
            args.manifests,
            fold=args.fold,
            seed=args.seed,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
    elif args.command == "prepare-rgb-challenge":
        report = prepare_rgb_challenge(
            args.archive,
            args.destination,
            args.manifests,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
    elif args.command == "train":
        path = run_training(
            load_config(args.config),
            device_name=args.device,
            resume=args.resume,
        )
        print(path)
    elif args.command == "train-quality":
        path = train_quality_gate(
            args.manifest,
            args.output,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device_name=args.device,
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
    elif args.command == "generate-ablations":
        paths = generate_b0_factor_ablations(
            args.base,
            args.output,
            args.results_root,
        )
        print("\n".join(str(path) for path in paths))
    elif args.command == "train-yolo":
        from weldvision.yolo_source import train_yolo_source

        path = train_yolo_source(
            load_config(args.config),
            device_name=args.device,
            resume=args.resume,
        )
        print(path)
    elif args.command == "evaluate-yolo":
        from weldvision.yolo_source import evaluate_yolo_source

        report = evaluate_yolo_source(
            load_config(args.config),
            args.checkpoint,
            split=args.split,
            device_name=args.device,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
    elif args.command == "explain-network":
        from weldvision.network_tour import write_network_tour

        path = write_network_tour(
            args.output,
            image_path=args.image,
            checkpoint=args.checkpoint,
            device_name=args.device,
            image_size=args.image_size,
            confidence=args.confidence,
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
