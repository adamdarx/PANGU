#!/usr/bin/env python3
"""Validate the EH-0 electron-module contract and KHARMA reference vectors."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from typing import Any


EXPECTED_MODELS = {"constant", "howes", "kawazura", "werner", "rowan", "sharma"}
REQUIRED_ASSEMBLIES = {
    ("cks", "dynamic"),
    ("mks", "static"),
    ("mks", "dynamic"),
    ("cks", "static"),
}
EXPECTED_TASK_ORDER = [
    "grmhd_riemann",
    "fofc_fluid_and_emf",
    "passive_scalar_flux_and_corner_emf",
    "parthenon_flux_correction",
    "fluid_rk_cell_and_face_update_and_independent_electron_rk_update",
    "fluid_sources_and_boundary_exchange",
    "grmhd_c2p_floors_ceilings_and_excision",
    "electron_u2p",
    "electron_heating_entire_domain",
    "electron_p2u",
    "timestep_and_refinement",
]


def close(actual: float, expected: float, name: str) -> None:
    if not math.isfinite(actual):
        raise RuntimeError(f"{name} is not finite: {actual}")
    if not math.isclose(actual, expected, rel_tol=5.0e-15, abs_tol=5.0e-15):
        raise RuntimeError(f"{name} mismatch: actual={actual:.17g} expected={expected:.17g}")


def calculate(case: dict[str, Any], constants: dict[str, float]) -> tuple[dict[str, float], dict[str, float]]:
    rho = float(case["input"]["rho"])
    internal = float(case["input"]["internal_energy"])
    magnetic2 = float(case["input"]["magnetic_b_squared"])
    kel = float(case["input"]["electron_entropy"])
    gamma = float(constants["gamma_gas"])
    gamma_e = float(constants["gamma_electron"])
    gamma_p = float(constants["gamma_proton"])
    mass_ratio = float(constants["proton_mass_cgs"]) / float(constants["electron_mass_cgs"])

    t_proton = (gamma_p - 1.0) * internal / rho
    t_electron = kel * rho ** (gamma_e - 1.0)
    temperature_ratio = t_proton / t_electron
    beta = min(2.0 * rho * t_proton / magnetic2, 1.0e20)

    log_ratio = math.log10(temperature_ratio)
    beta_exponent = 2.0 - 0.2 * log_ratio
    c2 = (1.6 if temperature_ratio <= 1.0 else 1.2) / temperature_ratio
    c3 = 18.0 + 5.0 * log_ratio if temperature_ratio <= 1.0 else 18.0
    beta_power = beta**beta_exponent
    qi_over_qe = (
        0.92
        * (c2 * c2 + beta_power)
        / (c3 * c3 + beta_power)
        * math.exp(-1.0 / beta)
        * math.sqrt(mass_ratio * temperature_ratio)
    )
    howes = 1.0 / (1.0 + qi_over_qe)

    qi_over_qe = 35.0 / (
        1.0 + (beta / 15.0) ** (-1.4) * math.exp(-0.1 / temperature_ratio)
    )
    kawazura = 1.0 / (1.0 + qi_over_qe)

    werner_sigma = magnetic2 / rho
    werner = 0.25 * (
        1.0 + math.sqrt((werner_sigma / 5.0) / (2.0 + werner_sigma / 5.0))
    )

    rowan_beta = 2.0 * (gamma_p - 1.0) * internal / magnetic2
    gas_pressure = (gamma - 1.0) * internal
    rowan_sigma = magnetic2 / (rho + internal + gas_pressure)
    rowan_beta_max = 0.25 / rowan_sigma
    rowan_base = 1.0 - rowan_beta / rowan_beta_max
    if rowan_base < 0.0:
        raise RuntimeError("EH-0 Rowan reference lies outside the non-negative fit domain")
    rowan = 0.5 * math.exp(
        -(rowan_base**3.3) / (1.0 + 1.2 * rowan_sigma**0.7)
    )

    qe_over_qi = 0.33 * math.sqrt(1.0 / temperature_ratio)
    sharma = 1.0 / (1.0 + 1.0 / qe_over_qi)

    derived = {
        "proton_temperature": t_proton,
        "electron_temperature": t_electron,
        "tp_over_te": temperature_ratio,
        "beta_i": beta,
        "werner_sigma": werner_sigma,
        "rowan_beta_i": rowan_beta,
        "rowan_sigma": rowan_sigma,
        "rowan_beta_max": rowan_beta_max,
    }
    fractions = {
        "constant": float(constants["constant_fraction"]),
        "howes": howes,
        "kawazura": kawazura,
        "werner": werner,
        "rowan": rowan,
        "sharma": sharma,
    }
    return derived, fractions


def validate_source_boundary(root: pathlib.Path, contract: dict[str, Any]) -> None:
    electron_root = root / "plugins" / "electron" / "electron"
    if not electron_root.exists():
        if contract["implementation_status"] != "contract_only":
            raise RuntimeError("electron implementation is missing but contract is not contract_only")
        return

    forbidden = contract["dependency_contract"]["forbidden_source_tokens"]
    violations: list[str] = []
    for source in sorted(electron_root.rglob("*")):
        if source.suffix not in {".cc", ".h"}:
            continue
        text = source.read_text(encoding="utf-8")
        for token in forbidden:
            if token in text:
                violations.append(f"{source.relative_to(root)} contains forbidden token {token!r}")
    if violations:
        raise RuntimeError("\n".join(violations))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=pathlib.Path)
    parser.add_argument("--contract", required=True, type=pathlib.Path)
    parser.add_argument("--reference", required=True, type=pathlib.Path)
    args = parser.parse_args()

    root = args.root.resolve()
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    reference = json.loads(args.reference.read_text(encoding="utf-8"))

    if contract["phase"] != "EH-0" or contract["module"] != "electrons":
        raise RuntimeError("unexpected electron contract identity")
    optionality = contract["runtime_optionality"]
    if optionality["canonical_key"] != "electrons/enabled":
        raise RuntimeError("electrons/enabled must remain the canonical activation key")
    for key in (
        "disabled_registers_package",
        "disabled_registers_fields",
        "disabled_adds_tasks",
        "disabled_changes_grmhd_kernel",
    ):
        if optionality[key] is not False:
            raise RuntimeError(f"runtime optionality contract violated: {key}")

    assemblies = {
        (entry["metric"], entry["mode"])
        for entry in contract["supported_assemblies"]
        if entry["required"]
    }
    if assemblies != REQUIRED_ASSEMBLIES:
        raise RuntimeError(f"required assembly matrix changed: {sorted(assemblies)}")
    cks_dynamic = next(
        entry
        for entry in contract["supported_assemblies"]
        if entry["metric"] == "cks" and entry["mode"] == "dynamic"
    )
    if "smr" not in cks_dynamic["tests"] or "custom_boundary" not in cks_dynamic["tests"]:
        raise RuntimeError("CKS/Dynamic must test both SMR and custom boundaries")

    if set(contract["models"]) != EXPECTED_MODELS:
        raise RuntimeError("electron heating model set differs from KHARMA contract")
    if set(contract["implemented_models"]) != EXPECTED_MODELS:
        raise RuntimeError("all six electron heating models must be implemented after EH-3")
    if contract["implementation_status"] != "dissipative_all_models":
        raise RuntimeError("electron implementation status is not the completed EH-3 state")
    if contract["task_order"] != EXPECTED_TASK_ORDER:
        raise RuntimeError("electron RK task order changed")
    if any(field["registered_when_disabled"] for field in contract["fields"]):
        raise RuntimeError("electron fields must not exist when the module is disabled")

    cmake = (root / "src" / "CMakeLists.txt").read_text(encoding="utf-8")
    for value in ('METRIC STREQUAL "cks"', 'METRIC STREQUAL "mks"', 'MODE STREQUAL "dynamic"', 'MODE STREQUAL "static"'):
        if value not in cmake:
            raise RuntimeError(f"compile-time Geometry assembly selector is missing {value}")

    validate_source_boundary(root, contract)

    if reference["source"]["commit"] != contract["reference"]["commit"]:
        raise RuntimeError("contract/reference KHARMA commits disagree")
    constants = reference["constants"]
    seen: set[str] = set()
    for case in reference["cases"]:
        if case["name"] in seen:
            raise RuntimeError(f"duplicate reference case {case['name']}")
        seen.add(case["name"])
        derived, fractions = calculate(case, constants)
        for name, actual in derived.items():
            close(actual, float(case["derived"][name]), f"{case['name']}.{name}")
        if set(case["fractions"]) != EXPECTED_MODELS:
            raise RuntimeError(f"{case['name']} has an incomplete model reference")
        for name, actual in fractions.items():
            close(actual, float(case["fractions"][name]), f"{case['name']}.{name}")
            if not 0.0 <= actual <= 1.0:
                raise RuntimeError(f"{case['name']}.{name} is outside [0,1]")

    print(
        "EH-0 electron contract PASS: runtime-optional package, four Geometry assemblies, "
        f"{len(reference['cases'])} KHARMA FP64 states, six heating models"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, RuntimeError, ValueError) as error:
        print(f"EH-0 electron contract FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
