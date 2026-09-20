#!/usr/bin/env python3
"""Reject deprecated or ambiguous PANGU-owned symbol spellings."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


BANNED_PATTERNS = {
    r"\bLAXF\b": "use LLF for local Lax--Friedrichs",
    r"\bHlle\b": "use HLLE",
    r"\bLlf\b": "use LLF",
    r"\bHllc\b": "use HLLC",
    r"\bHlld\b": "use HLLD",
    r"\bPlm\b": "use PLM",
    r"\bPpm\b": "use PPM",
    r"\bPpmc\b": "use PPMC",
    r"\bWenoZ\b": "use WENOZ in code and WENO-Z in prose",
    r"\bGrmhd\b": "use GRMHD",
    r"\bSrmhd\b": "use SRMHD",
    r"\bHLLEGRDirect\b": "use SolveGRMHDRiemannDirect with a typed solver",
    r"\bLocalLaxFriedrichs\b": "use a riemann::Solver template argument",
    r"\bRMHDState\b": "use RMHDFluxState",
    r"\bGRPhysicalState\b": "use BuildGRMHDFluxState",
    r"\bLLFGR\b": "use SolveGRMHDLLFFromStates",
    r"\b[A-Za-z_][A-Za-z0-9_]*TaskMesh\b": "write the execution level before Task",
    r"\bConservedToPrimitiveTaskMesh\b": "use ConservedToPrimitiveMeshTask",
    r"\bPrimitiveToConservedTaskMesh\b": "use PrimitiveToConservedMeshTask",
    r"\bAthenaKCellUpdate\b": "use UpdateCellConservedMeshTask",
    r"\bAthenaKFaceUpdate\b": "use UpdateFaceFieldsMeshTask",
    r"\bFluxFunction\b": "use BlockFluxTask",
    r"\bStageFunction\b": "use BlockSourceTask",
    r"\bCorrectionFunction\b": "use BlockFluxCorrectionTask",
    r"\bMasterFunction\b": "use a Compute...Residual name",
    r"\bMHDMasterFunction\b": "use Compute...Residual",
    r"\bBracketFunction\b": "use Compute...BracketResidual",
    r"\bCoordinateLightSpeeds\b": "use ComputeCoordinateLightSpeeds",
    r"\bComovingMagneticFieldSquared\b": "use ComputeComovingMagneticFieldSquared",
    r"\bPrimitiveToConserved(?:SR|GR|SRMHD|GRMHD)[A-Za-z0-9_]*\b": "use Convert...P2C",
    r"\bConservedToPrimitive(?:SR|GR|SRMHD|GRMHD)[A-Za-z0-9_]*\b": "use Solve...C2P",
    r"\bMHDPrimitive\b": "use MHDPrimitiveState",
}

DIRECT_ALLOWLIST = {"SolveGRMHDRiemannDirect"}
DIRECT_SYMBOL = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*Direct\b")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    source_root = args.root / "src"
    failures: list[str] = []
    for path in sorted((args.root / "cmake").glob("Pangu*.cmake")):
        failures.append(
            f"{path.relative_to(args.root)}: use the uppercase PANGU project prefix"
        )
    production_roots = (source_root, args.root / "plugins")
    for path in sorted(path for tree in production_roots for path in tree.rglob("*")):
        if path.suffix not in {".cc", ".h"}:
            continue
        text = path.read_text(encoding="utf-8")
        for pattern, reason in BANNED_PATTERNS.items():
            for match in re.finditer(pattern, text):
                line = text.count("\n", 0, match.start()) + 1
                failures.append(f"{path.relative_to(args.root)}:{line}: {match.group(0)}: {reason}")
        for match in DIRECT_SYMBOL.finditer(text):
            symbol = match.group(0)
            if symbol not in DIRECT_ALLOWLIST:
                line = text.count("\n", 0, match.start()) + 1
                failures.append(
                    f"{path.relative_to(args.root)}:{line}: {symbol}: "
                    "register and document every Direct implementation"
                )

    if failures:
        print("PANGU naming convention audit failed:")
        print("\n".join(failures))
        return 1
    print("PANGU naming convention audit passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
