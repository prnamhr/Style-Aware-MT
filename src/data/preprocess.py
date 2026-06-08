import argparse
import csv
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple

# Increase CSV field size limit for large paragraph blocks
csv.field_size_limit(sys.maxsize)

# --- Configuration & Regex ---
ARABIC_SCRIPT_RE = re.compile(
    r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]"
)
LATIN_SCRIPT_RE = re.compile(r"[A-Za-z]")

PERSIAN_STANDARDIZATION_MAP = str.maketrans({"ي": "ی", "ك": "ک", "ـ": ""})

ENGLISH_PUNCTUATION_MAP = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'", "…": "..."})

# --- Normalization Functions ---


def normalize_text(text: str, is_source: bool = True) -> str:
    if not text:
        return ""

    # 1. Unicode Normalization
    text = unicodedata.normalize("NFKC", str(text))

    # 2. Remove invisible chars
    text = text.replace("\ufeff", "").replace("\u200b", "")

    # 3. Collapse newlines to single space (keeps TSV output single-line)
    text = text.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")

    if is_source:
        text = text.translate(PERSIAN_STANDARDIZATION_MAP)
    else:
        text = text.translate(ENGLISH_PUNCTUATION_MAP)
        # Fix spacing around punctuation
        text = re.sub(r"\s+([,.;:!?])", r"\1", text)
        text = re.sub(r"\(\s+", "(", text)
        text = re.sub(r"\s+\)", ")", text)

    # 4. Final whitespace cleanup
    text = re.sub(r"\s+", " ", text).strip()
    return text


def validate_record(source: str, target: str, min_ar: int, min_en: int) -> List[str]:
    reasons = []
    if not source or not target:
        reasons.append("EmptyField")
    if source == target:
        reasons.append("SourceEqualsTarget")
    if len(ARABIC_SCRIPT_RE.findall(source)) < min_ar:
        reasons.append("LowArabicScriptCount")
    if len(LATIN_SCRIPT_RE.findall(target)) < min_en:
        reasons.append("LowLatinScriptCount")
    if "\ufffd" in source or "\ufffd" in target:
        reasons.append("EncodingErrorChar")
    return reasons


# --- Parsing Logic ---


def process_file(
    input_path: Path,
    unit_type: str,
    has_header: bool,
    col_mapping: Dict[str, int],
    min_ar: int,
    min_en: int,
) -> Tuple[List[Dict], List[Dict]]:
    cleaned = []
    dropped = []
    seen_hashes = set()

    print(f"Processing {unit_type} from: {input_path.name}...")

    with input_path.open("r", encoding="utf-8-sig", newline="") as f:
        # csv.reader handles quotes and newlines automatically
        reader = csv.reader(f, delimiter="\t", quotechar='"')

        if has_header:
            next(reader, None)  # Skip header row

        for line_num, row in enumerate(reader, start=1):
            if not row:
                continue

            try:
                raw_src = row[col_mapping["src"]] if len(row) > col_mapping["src"] else ""
                raw_tgt = row[col_mapping["tgt"]] if len(row) > col_mapping["tgt"] else ""
                raw_meta = row[col_mapping["meta"]] if len(row) > col_mapping["meta"] else ""

                # Paragraph index
                p_idx = (
                    row[col_mapping["pid"]]
                    if "pid" in col_mapping and len(row) > col_mapping["pid"]
                    else None
                )
                # Sentence index (only for sentence file)
                s_idx = (
                    row[col_mapping["sid"]]
                    if "sid" in col_mapping and len(row) > col_mapping["sid"]
                    else None
                )

            except IndexError:
                dropped.append({"line": line_num, "reason": "ColumnMismatch", "raw": str(row)})
                continue

            # Normalize
            norm_src = normalize_text(raw_src, is_source=True)
            norm_tgt = normalize_text(raw_tgt, is_source=False)
            norm_meta = normalize_text(raw_meta, is_source=True)

            # Validate
            reasons = validate_record(norm_src, norm_tgt, min_ar, min_en)

            if not reasons:
                # Deduplication check
                row_hash = (norm_src, norm_tgt)
                if row_hash in seen_hashes:
                    reasons.append("Duplicate")
                else:
                    seen_hashes.add(row_hash)

            # Store result
            record = {
                "unit_type": unit_type,
                "source_text": norm_src,
                "target_text": norm_tgt,
                "source_name": norm_meta,
                "paragraph_id": p_idx,
                "sentence_id": s_idx,
            }

            if reasons:
                record["rejection_reasons"] = ";".join(reasons)
                dropped.append(record)
            else:
                cleaned.append(record)

    return cleaned, dropped


def write_outputs(output_dir: Path, filename_stem: str, cleaned: List[Dict], dropped: List[Dict]):
    # Write Cleaned TSV
    clean_path = output_dir / f"{filename_stem}_cleaned.tsv"
    fieldnames = ["source_text", "target_text", "source_name", "paragraph_id", "sentence_id"]

    with clean_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in cleaned:
            out_row = {k: row.get(k) for k in fieldnames}
            writer.writerow(out_row)

    # Write Dropped TSV
    drop_path = output_dir / f"{filename_stem}_dropped.tsv"
    with drop_path.open("w", encoding="utf-8", newline="") as f:
        if dropped:
            drop_fields: List[str] = []
            for rec in dropped:
                for k in rec:
                    if k not in drop_fields:
                        drop_fields.append(k)
            writer = csv.DictWriter(
                f, fieldnames=drop_fields, delimiter="\t", restval="", extrasaction="ignore"
            )
            writer.writeheader()
            writer.writerows(dropped)

    # Write JSONL for LLM training
    jsonl_path = output_dir / f"{filename_stem}_cleaned.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in cleaned:
            obj = {
                "input": row["source_text"],
                "output": row["target_text"],
                "metadata": {"source": row["source_name"], "type": row["unit_type"]},
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# --- Main Execution ---


def main():
    parser = argparse.ArgumentParser(
        description="Normalize, validate, deduplicate, and convert raw TSVs."
    )
    parser.add_argument("--sentence_tsv", type=str, default="data/raw/sentence_pairs.tsv")
    parser.add_argument("--paragraph_tsv", type=str, default="data/raw/clean_paragraph_pairs.tsv")
    parser.add_argument("--output_dir", type=str, default="data/processed")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Process Sentence File — columns: src | tgt | name | para_id | sent_id (no header)
    sent_cols = {"src": 0, "tgt": 1, "meta": 2, "pid": 3, "sid": 4}

    s_clean, s_drop = process_file(
        Path(args.sentence_tsv),
        "sentence",
        has_header=False,
        col_mapping=sent_cols,
        min_ar=1,
        min_en=1,
    )
    write_outputs(out_dir, "sentences", s_clean, s_drop)

    # 2. Process Paragraph File — columns: text | translation | source | paragraph (has header)
    para_cols = {"src": 0, "tgt": 1, "meta": 2, "pid": 3}

    p_clean, p_drop = process_file(
        Path(args.paragraph_tsv),
        "paragraph",
        has_header=True,
        col_mapping=para_cols,
        min_ar=1,
        min_en=1,
    )
    write_outputs(out_dir, "paragraphs", p_clean, p_drop)

    print("\n--- Summary ---")
    print(f"Sentences: {len(s_clean)} kept, {len(s_drop)} dropped.")
    print(f"Paragraphs: {len(p_clean)} kept, {len(p_drop)} dropped.")
    print(f"Outputs written to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
