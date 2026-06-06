import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser(description="Split cleaned sentence data into train/val/test (80/10/10).")
    parser.add_argument("--input_file", type=str, default="data/processed/sentences_cleaned.jsonl")
    parser.add_argument("--output_dir", type=str, default="data/splits")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_file = Path(args.input_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {input_file}...")
    df = pd.read_json(input_file, lines=True)

    train_val, test = train_test_split(df, test_size=0.10, random_state=args.seed, shuffle=True)
    # 0.1111 of the 90% remainder ≈ 10% of total
    train, val = train_test_split(train_val, test_size=0.1111, random_state=args.seed, shuffle=True)

    print(f"Saving splits to {output_dir}...")
    train.to_json(output_dir / "train.jsonl", orient="records", lines=True, force_ascii=False)
    val.to_json(output_dir / "val.jsonl", orient="records", lines=True, force_ascii=False)
    test.to_json(output_dir / "test.jsonl", orient="records", lines=True, force_ascii=False)

    print(f"Done!\n  Train: {len(train)}\n  Val:   {len(val)}\n  Test:  {len(test)}")


if __name__ == "__main__":
    main()
