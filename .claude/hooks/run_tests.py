"""PostToolUse hook: run pytest after Claude edits a Python file or a level in this project.

Silent when the tests pass. On failure it exits with code 2 so Claude Code feeds the
pytest output back to Claude.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    try:
        data = json.load(sys.stdin)
    except ValueError:
        return 0
    raw = (data.get("tool_input") or {}).get("file_path") or (data.get("tool_response") or {}).get("filePath")
    if not raw:
        return 0
    path = Path(raw).resolve()
    if ROOT not in path.parents or ".venv" in path.parts:
        return 0
    if not (path.suffix == ".py" or (path.suffix == ".json" and "levels" in path.parts)):
        return 0

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-x", "--no-header", "-p", "no:cacheprovider"],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=110,
        env={**os.environ, "SDL_VIDEODRIVER": "dummy"},
    )
    if result.returncode == 0:
        return 0
    tail = "\n".join((result.stdout + result.stderr).strip().splitlines()[-40:])
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    print(f"Tests failed after editing {path.relative_to(ROOT)}:\n{tail}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
