#!/usr/bin/env python3
"""Clopper-Pearson bounds on P(judge says Yes) from an attack_result JSON.
Streams the file with ijson so the full JSON is not loaded.
Usage: python attack_result_clopper_pearson.py attack_result/<file>.json
"""
import sys

try:
    import ijson
except ImportError:
    ijson = None

from statsmodels.stats.proportion import proportion_confint


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if not path:
        print("Usage: python attack_result_clopper_pearson.py <attack_result.json>", file=sys.stderr)
        sys.exit(1)

    count_yes = 0
    total = 0

    if ijson:
        with open(path, "rb") as f:
            for attempt in ijson.items(f, "data.item.attempts.item"):
                total += 1
                if attempt.get("final_judge") == "Yes":
                    count_yes += 1
    else:
        import json
        with open(path) as f:
            data = json.load(f)
        for item in data.get("data", []):
            for attempt in item.get("attempts", []):
                total += 1
                if attempt.get("final_judge") == "Yes":
                    count_yes += 1

    if total == 0:
        print("No attempts found.", file=sys.stderr)
        sys.exit(1)

    low, high = proportion_confint(count_yes, total, alpha=0.05, method="beta")
    p = count_yes / total
    print(f"File: {path}")
    print(f"Judge Yes: {count_yes} / {total} = {p:.4f}")
    print(f"Clopper-Pearson 95% CI: [{low:.4f}, {high:.4f}]")


if __name__ == "__main__":
    main()
