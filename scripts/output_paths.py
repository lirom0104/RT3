"""Find retained outputs referenced by historical run records."""
from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"


def resolve_recorded_output(path):
    """Resolve an existing result using the directory migration table."""
    original = Path(path)
    if not original.is_absolute():
        original = ROOT / original
    if original.exists():
        return original
    try:
        relative = original.relative_to(OUTPUTS).as_posix()
    except ValueError:
        return original
    mapping = OUTPUTS / "PATH_MAPPING.csv"
    if not mapping.exists():
        return original
    with mapping.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.reader(stream))[1:]
    for old, new, *_ in sorted(rows, key=lambda row: len(row[0]), reverse=True):
        if relative != old and not relative.startswith(old + "/"):
            continue
        target = Path(new)
        if not target.is_absolute():
            target = OUTPUTS / target
        suffix = relative[len(old):].lstrip("/")
        if suffix:
            target = target / suffix
        if target.exists():
            return target
    return original
