#!/usr/bin/env python3
"""Repair Jupyter notebook outputs for GitHub / nbformat validation.

GitHub's renderer requires (among other things):
  - display_data / execute_result outputs: ``metadata`` (may be empty {})
  - stream outputs: ``name`` (stdout|stderr)
  - cells: ``metadata``; and ``id`` when nbformat_minor >= 5
  - top-level kernelspec / language_info: ``name``

Jupyter often omits these when re-saving outputs, which surfaces as:
  Invalid Notebook: 'metadata' is a required property

Usage:
  python scripts/fix_notebook_nbformat.py                 # all notebooks/
  python scripts/fix_notebook_nbformat.py path/a.ipynb ...
  python scripts/fix_notebook_nbformat.py --check          # exit 1 if any need fixes
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path


def _default_notebooks(root: Path) -> list[Path]:
    return sorted((root / "notebooks").glob("*.ipynb"))


def fix_notebook(path: Path, *, check_only: bool = False) -> bool:
    """Return True if the notebook needed (or would need) changes."""
    text = path.read_text(encoding="utf-8")
    nb = json.loads(text)
    changed = False

    md = nb.setdefault("metadata", {})
    if not isinstance(md, dict):
        nb["metadata"] = {}
        md = nb["metadata"]
        changed = True

    ks = md.setdefault("kernelspec", {})
    if not isinstance(ks, dict):
        md["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        changed = True
    else:
        for key, default in (("name", "python3"), ("display_name", "Python 3"), ("language", "python")):
            if key not in ks:
                ks[key] = default
                changed = True

    li = md.setdefault("language_info", {})
    if not isinstance(li, dict):
        md["language_info"] = {"name": "python"}
        changed = True
    elif "name" not in li:
        li["name"] = "python"
        changed = True

    need_ids = int(nb.get("nbformat_minor") or 0) >= 5

    for cell in nb.get("cells") or []:
        if "metadata" not in cell or not isinstance(cell.get("metadata"), dict):
            cell["metadata"] = cell["metadata"] if isinstance(cell.get("metadata"), dict) else {}
            changed = True
        if need_ids and "id" not in cell:
            cell["id"] = uuid.uuid4().hex[:8]
            changed = True
        for out in cell.get("outputs") or []:
            ot = out.get("output_type")
            if ot in ("display_data", "execute_result"):
                if "metadata" not in out or not isinstance(out.get("metadata"), dict):
                    out["metadata"] = {}
                    changed = True
            elif ot == "stream":
                if "name" not in out:
                    out["name"] = "stdout"
                    changed = True

    if changed and not check_only:
        path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return changed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="*", type=Path, help="Notebook paths (default: notebooks/*.ipynb)")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Do not write; exit 1 if any notebook would be modified",
    )
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[1]
    paths = list(args.paths) if args.paths else _default_notebooks(root)
    if not paths:
        print("No notebooks found.", file=sys.stderr)
        return 1

    dirty: list[Path] = []
    for path in paths:
        if not path.exists():
            print(f"missing: {path}", file=sys.stderr)
            return 1
        if path.suffix != ".ipynb":
            continue
        if fix_notebook(path, check_only=args.check):
            dirty.append(path)

    if dirty:
        verb = "would fix" if args.check else "fixed"
        for path in dirty:
            print(f"{verb}: {path}")
        return 1 if args.check else 0

    print(f"ok: {len(paths)} notebook(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
