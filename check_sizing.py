"""Exercise the sizing example through its command-line interface."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--repo", type=Path, required=True)
args = parser.parse_args()
cli = [
    sys.executable,
    str(args.repo / "examples/disaggregated/disaggregated_encoder/size_epd.py"),
]
examples = [
    ((8, "57", "14.3"), (2, 6, 85.8)),
    ((2, "7", "7"), (1, 1, 7)),
    ((4, "14", "7"), (1, 3, 14)),
    ((4, "11.2545", "6.7512"), (2, 2, 13.5024)),
]
for (gpus, encoder, pd), expected in examples:
    result = json.loads(subprocess.check_output(
        cli + ["--num-gpus", str(gpus), "--encoder-rps", encoder, "--pd-rps", pd]
    ))
    assert (result["encoder_gpus"], result["pd_gpus"], result["predicted_capacity_rps"]) == expected

invalid = [
    (1, "7", "7"), (4, "0", "7"), (4, "-1", "7"),
    (4, "nan", "7"), (4, "7", "inf"), (4, "1e-400", "7"),
    (4, "1e308", "7"), (10**400, "7", "7"),
]
for gpus, encoder, pd in invalid:
    result = subprocess.run(
        cli + ["--num-gpus", str(gpus), "--encoder-rps", encoder, "--pd-rps", pd],
        capture_output=True, text=True,
    )
    assert result.returncode == 2 and "Traceback" not in result.stderr, result.stderr
print("PASS: four allocations, including a tied optimum, and eight invalid inputs")
