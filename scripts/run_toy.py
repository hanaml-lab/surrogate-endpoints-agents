"""Run the CPU toy end to end (python scripts/run_toy.py [--config configs/toy.yaml]); same as python -m tenor.demo."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tenor.demo import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
