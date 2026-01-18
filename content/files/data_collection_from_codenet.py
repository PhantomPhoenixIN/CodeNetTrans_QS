#!/usr/bin/env python3
"""
Collect exactly ONE accepted solution per problem per language
from Project CodeNet, with problem-level splits per language.

Output (incremental JSONL, deduplicated across runs):
  /home/swaminathanj/TransRectify_New/content/datasets/codenet_single_solution.jsonl

FEATURES:
  ✔ Incremental JSONL writing
  ✔ Cross-run deduplication
  ✔ One accepted solution per problem-language
  ✔ Deterministic problem-level split per language
"""

import os
import json
import hashlib
import pandas as pd
from collections import Counter, defaultdict


# ============================================================
# Helpers
# ============================================================
def read_submission_file(lang_dir, submission_id, filename_ext):
    ext = filename_ext.lstrip(".")
    path = os.path.join(lang_dir, f"{submission_id}.{ext}")

    if not os.path.exists(path):
        alt = os.path.join(lang_dir, submission_id)
        if os.path.exists(alt):
            path = alt
        else:
            return None

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read().strip()
    except Exception:
        return None


def assign_split(problem_id, language):
    """
    Deterministic problem-level split per language.
    """
    key = f"{problem_id}::{language}"
    h = int(hashlib.md5(key.encode()).hexdigest(), 16) % 100

    if h < 80:
        return "train"
    elif h < 90:
        return "validation"
    else:
        return "test"


# ============================================================
# Main extractor
# ============================================================
def collect_input_codes(
    data_dir,
    metadata_dir,
    output_file,
    selected_languages=None,
    flush_every=50
):
    if selected_languages is None:
        selected_languages = []

    # --------------------------------------------------------
    # Load existing JSONL entries for dedupe
    # --------------------------------------------------------
    seen_keys = set()
    if os.path.exists(output_file):
        print(f"[INFO] Loading existing entries from {output_file}...")
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    seen_keys.add(f"{obj['problem_id']}::{obj['input_language']}")
                except Exception:
                    continue
        print(f"[INFO] Loaded {len(seen_keys)} existing entries for dedupe.\n")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    out_f = open(output_file, "a", encoding="utf-8")

    entries_written = 0
    problems_seen = 0
    lang_instance_counter = Counter()
    split_counter = Counter()

    # --------------------------------------------------------
    # Iterate through problems
    # --------------------------------------------------------
    for problem_id in sorted(os.listdir(data_dir)):
        problem_path = os.path.join(data_dir, problem_id)
        if not os.path.isdir(problem_path):
            continue

        meta_file = os.path.join(metadata_dir, f"{problem_id}.csv")
        if not os.path.exists(meta_file):
            continue

        try:
            df_meta = pd.read_csv(meta_file, dtype=str)
        except Exception:
            continue

        for lang in sorted(os.listdir(problem_path)):
            lang_dir = os.path.join(problem_path, lang)
            if not os.path.isdir(lang_dir):
                continue

            if selected_languages and lang not in selected_languages:
                continue

            unique_key = f"{problem_id}::{lang}"
            if unique_key in seen_keys:
                continue

            accepted = df_meta[
                (df_meta["language"] == lang) &
                (df_meta["status"] == "Accepted")
            ]

            if accepted.empty:
                continue

            code = None
            for _, row in accepted.iterrows():
                submission_id = str(row.get("submission_id", "")).strip()
                filename_ext = str(row.get("filename_ext", "")).strip()
                if not submission_id:
                    continue

                code = read_submission_file(lang_dir, submission_id, filename_ext)
                if code:
                    break

            if not code:
                continue

            split = assign_split(problem_id, lang)
            split_counter[split] += 1

            result = {
                "problem_id": problem_id,
                "input_language": lang,
                "output_language": "Java",
                "source_code": code,
                "translated_java_code": None,
                "split": split
            }

            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            seen_keys.add(unique_key)
            entries_written += 1
            lang_instance_counter[lang] += 1

            if entries_written % flush_every == 0:
                out_f.flush()
                try:
                    os.fsync(out_f.fileno())
                except Exception:
                    pass
                print(f"[INFO] Flushed {entries_written} entries...")

        problems_seen += 1
        if problems_seen % 200 == 0:
            print(f"[INFO] Processed {problems_seen} problems...")

    out_f.close()

    # ============================================================
    # Summary
    # ============================================================
    print("\n================ FINAL SUMMARY ================\n")
    print(f"Problems scanned : {problems_seen}")
    print(f"Entries written  : {entries_written}\n")

    print("### Split Distribution ###")
    for k in ["train", "validation", "test"]:
        print(f"{k:10s} : {split_counter[k]}")

    print("\n### Problems per Language ###")
    for lang, count in lang_instance_counter.most_common():
        print(f"{lang:10s} : {count}")

    print("\n==============================================\n")

    return {
        "entries_written": entries_written,
        "split_distribution": dict(split_counter),
        "per_language_counts": dict(lang_instance_counter)
    }


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    DATA_DIR = "/home/swaminathanj/TransRectify_New/content/dataset/Project_CodeNet/data"
    META_DIR = "/home/swaminathanj/TransRectify_New/content/dataset/Project_CodeNet/metadata"
    OUT_FILE = "/home/swaminathanj/TransRectify_New/content/datasets/codenet_single_solution.jsonl"

    selected_languages = [
        "Python", "C++", "C#", "C", "Kotlin", "Haxe", "Ruby", "Swift"
    ]

    summary = collect_input_codes(
        data_dir=DATA_DIR,
        metadata_dir=META_DIR,
        output_file=OUT_FILE,
        selected_languages=selected_languages,
        flush_every=50
    )