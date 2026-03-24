import argparse
import json
from collections import defaultdict
from pathlib import Path

from src.data.build_parallel_dataset import make_splits
from src.utils.io import write_json, write_jsonl
from src.utils.labeling import LABELS_6

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--translated_json", type=str, required=True, help="Path to the JSON file translated by the user")
    ap.add_argument("--out_dir", type=str, default="artifacts/data")
    ap.add_argument("--lr_lang", type=str, default="cy")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from src.utils.io import read_jsonl
    data = list(read_jsonl(args.translated_json))

    # Validate and format for parallel.jsonl
    examples = []
    for r in data:
        # Check required keys
        for k in ["id", "label_id", "label", "text_en", "text_hi", f"text_{args.lr_lang}"]:
            if k not in r:
                raise ValueError(f"Missing required key '{k}' in row {r.get('id', 'unknown')}")

        y_id = int(r["label_id"])
        examples.append({
            "id": r["id"],
            "label_id": y_id,
            "label": r["label"],
            "labels_onehot": [1 if k == y_id else 0 for k in range(len(LABELS_6))],
            "domain": str(r.get("domain", "")),
            "text_en": str(r["text_en"]),
            "text_hi": str(r["text_hi"]),
            f"text_{args.lr_lang}": str(r[f"text_{args.lr_lang}"]),
        })

    # Create splits
    ids = [e["id"] for e in examples]
    counts = defaultdict(int)
    for e in examples:
        counts[e["label"]] += 1

    splits = make_splits(ids, [e["label_id"] for e in examples], seed=args.seed)
    
    write_json(out_dir / "splits.json", splits)
    write_jsonl(out_dir / "parallel.jsonl", examples)

    summary = {"total": len(examples), "per_label": dict(sorted(counts.items()))}
    write_json(out_dir / "summary.json", summary)
    
    print(f"[import_translated] Processed {len(examples)} manually translated examples.")
    print(f"  - {out_dir / 'parallel.jsonl'}")
    print(f"  - {out_dir / 'splits.json'}")
    print(f"  - {out_dir / 'summary.json'}")
    print("[import_translated] Label distribution:")
    for k, v in summary["per_label"].items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
