"""CLI entry point that validates raw folders and creates a dataset manifest."""

import argparse
from pathlib import Path

from app.manifest import write_raw_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect PlantVillage Train/Val/Test folders.")
    parser.add_argument("--data-dir", type=Path,
                        default=Path("data/Crop Disease Detection Dataset/Plant Village Dataset"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/dataset_manifest.json"))
    args = parser.parse_args()
    manifest = write_raw_manifest(args.data_dir, args.output)
    print(
        f"Validated {manifest['num_classes']} classes: {manifest['train_images']} train / "
        f"{manifest['validation_images']} validation / {manifest['test_images']} test images"
    )
    if manifest["empty_train_classes"]:
        print(
            "WARNING: training classes without accessible image files: "
            f"{', '.join(manifest['empty_train_classes'])}"
        )
    print(f"Manifest written to {args.output}")


if __name__ == "__main__":
    main()
