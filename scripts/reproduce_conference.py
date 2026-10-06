"""Regenerate the conference Exp5 evidence and table from saved summaries."""
import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFERENCE = ROOT / "conference"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "generated/conference")
    args = parser.parse_args()
    output = args.output.resolve()
    if CONFERENCE == output or CONFERENCE in output.parents:
        parser.error("Choose an output outside the archived conference evidence.")
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(CONFERENCE), MPLBACKEND="Agg", MPLCONFIGDIR=str(output / ".matplotlib"))
    base = [sys.executable, "-m", "cognition.expanded_dataset_phase042"]
    subprocess.run(base + ["evidence-analysis", "--output-root", str(output)],
                   cwd=CONFERENCE, env=env, check=True)
    subprocess.run(base + ["final-outputs", "--evidence-analysis", str(output / "evidence_analysis/exp5_evidence_analysis.json"),
                          "--output-root", str(output)], cwd=CONFERENCE, env=env, check=True)
    name = "corrected_expanded_table_rows.csv"
    with (output / "final_outputs_20260522" / name).open() as f:
        actual = list(csv.DictReader(f))
    with (CONFERENCE / "results/exp5/final_outputs_20260522" / name).open() as f:
        expected = list(csv.DictReader(f))
    metadata_fields = {"source_artifact_path"}
    scientific = lambda rows: [{k: v for k, v in row.items() if k not in metadata_fields} for row in rows]
    if scientific(actual) != scientific(expected):
        raise ValueError("Regenerated conference Exp5 table differs from the archived table")
    result = {"conference_exp5_table_matches": True, "rows": len(actual), "model_requests_made": 0}
    (output / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
