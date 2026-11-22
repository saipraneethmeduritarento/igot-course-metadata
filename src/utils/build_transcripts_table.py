"""Build a flat CSV table of resource transcripts from the combined transcripts JSON.

Input is the file written by combine_transcripts.py (one record per Live resource).
Output has one row per resource with the columns in COLUMNS.

    python -m src.utils.build_transcripts_table [input.json] [output.csv]
"""

import csv
import json
import os
import sys

# (CSV column, key in the combined JSON)
COLUMNS = [
    ("course_id", "course_id"),
    ("course_name", "course_name"),
    ("resource_id", "resource_id"),
    ("resource_name", "name"),
    ("mime_type", "mimeType"),
    ("audio_transcript", "audio_transcript"),
    ("video_transcript", "video_transcript"),
]


def build_transcripts_table(json_path, csv_path):
    """Write the selected columns of every record in json_path to csv_path. Returns the row count."""
    with open(json_path, encoding="utf-8") as f:
        records = json.load(f)

    # Group a course's resources together.
    records.sort(key=lambda r: (r.get("course_id") or "", r.get("resource_id") or ""))

    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    # utf-8-sig so Excel shows Hindi and other non-Latin text correctly.
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([col for col, _ in COLUMNS])
        for r in records:
            writer.writerow(["" if r.get(key) is None else r[key] for _, key in COLUMNS])
    return len(records)


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "data/input/resource_transcripts.json"
    dst = sys.argv[2] if len(sys.argv) > 2 else "data/input/resource_transcripts_table.csv"
    n = build_transcripts_table(src, dst)
    print(f"Wrote {n} rows to {dst}")
