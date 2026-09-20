#!/usr/bin/env python3
"""Enforce stage-1 ownership and framework-boundary invariants."""

from __future__ import annotations

import argparse
import pathlib
import re


def production_text(text: str) -> str:
    """Return code and literals while excluding source-attribution comments."""
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    return re.sub(r"//.*", "", text)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=pathlib.Path)
    root = parser.parse_args().root.resolve()
    source_root = root / "src"
    input_root = root / "input"

    if not source_root.is_dir() or not input_root.is_dir():
        raise RuntimeError("PANGU requires src/ and singular input/ directories")
    if (root / "inputs").exists():
        raise RuntimeError("legacy plural inputs/ directory must not be a PANGU interface")

    banned = {
        "athenak/": "PANGU source must not include AthenaK implementation files",
        "MPI_Isend(": "point-to-point communication belongs to Parthenon",
        "MPI_Irecv(": "point-to-point communication belongs to Parthenon",
        "MPI_Send(": "point-to-point communication belongs to Parthenon",
        "MPI_Recv(": "point-to-point communication belongs to Parthenon",
    }
    violations: list[str] = []
    production_roots = (source_root, root / "plugins")
    for path in sorted(path for tree in production_roots for path in tree.rglob("*")):
        if path.suffix not in {".cc", ".h", ".in"}:
            continue
        text = path.read_text(encoding="utf-8")
        for token, reason in banned.items():
            if token in text:
                violations.append(f"{path.relative_to(root)}: {reason} ({token})")
        active = production_text(text)
        for token in ("AthenaK", "AthenaPK", "athenak_ppm", "athenapk_ppm"):
            if token in active:
                violations.append(
                    f"{path.relative_to(root)}: production code contains external-framework "
                    f"compatibility symbol ({token})"
                )
    if violations:
        raise RuntimeError("\n".join(violations))

    driver_text = production_text((source_root / "driver" / "driver.cc").read_text(encoding="utf-8"))
    for module in ("hydro", "mhd", "electron", "radiation", "z4c", "numerical_relativity"):
        if module in driver_text.lower():
            raise RuntimeError(f"driver.cc must not identify the {module} physics module")

    inputs = list(input_root.rglob("*.in"))
    if not inputs:
        raise RuntimeError("input/ contains no versioned PANGU decks")
    print(f"source-layout audit PASS: {len(inputs)} input decks, no banned framework coupling")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
