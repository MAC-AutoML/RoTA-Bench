"""Create the CPU demo's small task catalog from the full benchmark JSONL."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
SOURCE = REPO_ROOT / "data" / "tasks_674.jsonl"
SPLIT = ROOT / "tasks" / "test.json"
OUTPUT = ROOT / "tasks" / "tasks.jsonl"


def main() -> None:
    wanted = set(json.loads(SPLIT.read_text(encoding="utf-8"))["sample_ids"])
    remaining = set(wanted)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with SOURCE.open("r", encoding="utf-8") as source, OUTPUT.open("w", encoding="utf-8", newline="\n") as output:
        for line in source:
            if not remaining:
                break
            if any(f'"sample_id": "{sample_id}"' in line for sample_id in remaining):
                item = json.loads(line)
                sample_id = item.get("sample_id")
                if sample_id in remaining:
                    output.write(line if line.endswith("\n") else line + "\n")
                    remaining.remove(sample_id)
    if remaining:
        OUTPUT.unlink(missing_ok=True)
        raise SystemExit(f"Missing test tasks: {sorted(remaining)}")
    print(f"Prepared {len(wanted)} tasks at {OUTPUT} ({OUTPUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
