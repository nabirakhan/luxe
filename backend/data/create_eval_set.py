"""One-time script: sample a fixed 20-image eval set from DeepFashion train split.

Usage:
    python data/create_eval_set.py --partition /content/deepfashion/list_eval_partition.txt

Writes data/eval_filenames.txt. Run once, commit the output file, never change it.
eval_runner.py reads eval_filenames.txt to load the fixed test set.
"""

import argparse
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--partition",
        default="/content/deepfashion/list_eval_partition.txt",
        help="Path to list_eval_partition.txt",
    )
    parser.add_argument("--n", type=int, default=20, help="Number of images to sample")
    parser.add_argument(
        "--out",
        default=str(Path(__file__).parent / "eval_filenames.txt"),
        help="Output path",
    )
    args = parser.parse_args()

    # Parse train images
    train_images = []
    with open(args.partition) as f:
        lines = f.readlines()[2:]  # skip two header lines
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        img_name, status = parts[0], parts[1]
        if status == "train":
            train_images.append(img_name)

    if len(train_images) < args.n:
        raise ValueError(f"Only {len(train_images)} train images found, need {args.n}")

    random.seed(42)
    selected = random.sample(train_images, args.n)
    selected.sort()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for name in selected:
            f.write(name + "\n")

    print(f"Wrote {len(selected)} filenames to {out_path}")


if __name__ == "__main__":
    main()
