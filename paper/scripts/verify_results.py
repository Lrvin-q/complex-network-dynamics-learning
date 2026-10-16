#!/usr/bin/env python3
"""验证论文使用的关键数值、来源哈希与最终选择规则。"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def close(a: float, b: float, tol: float = 5e-7) -> None:
    if not math.isclose(a, b, rel_tol=0, abs_tol=tol):
        raise AssertionError(f"{a} != {b}")


def main() -> None:
    manifest = json.loads((PAPER / "metrics_manifest.json").read_text(encoding="utf-8"))
    results_tex = (PAPER / "sections" / "06_results.tex").read_text(encoding="utf-8")
    appendix_tex = (PAPER / "sections" / "09_appendix.tex").read_text(encoding="utf-8")
    for source in manifest["sources"]:
        path = ROOT / source["path"]
        if digest(path) != source["sha256"]:
            raise AssertionError(f"source changed: {source['path']}")

    expected_known = {
        "ER": (0.3035925510, 0.0111221508, 0.1055009218, 0.0106194510),
        "WS": (0.3079190919, 0.0137394997, 0.1003556976, 0.0063151373),
        "BA": (0.3006773018, 0.0140302180, 0.1011585193, 0.0067465098),
    }
    rows = manifest["known_topology"]["twob3l_bi"]["rows"]
    for row in rows:
        beta, perk, gamma, mae_i = expected_known[row["network"]]
        close(row["beta_hat"], beta)
        close(row["per_k_mae"], perk)
        close(row["gamma_hat"], gamma)
        close(row["autoreg_mae_i"], mae_i)

    expected_seeds = {"ER": 42, "WS": 7, "BA": 7}
    for net, expected_seed in expected_seeds.items():
        selected = manifest["unknown_topology_selected"][net]
        if selected["seed"] != expected_seed:
            raise AssertionError(f"wrong selected seed for {net}")
        candidates = [r for r in manifest["unknown_topology_all_runs"] if r["network"] == net]
        best = min(candidates, key=lambda r: r["validation_ce"])
        if best["seed"] != expected_seed:
            raise AssertionError(f"selection is not minimum validation CE for {net}")

    expected_edges = {"ER": (528, 511), "WS": (300, 300), "BA": (295, 291)}
    for net, edges in expected_edges.items():
        selected = manifest["unknown_topology_selected"][net]
        if (selected["predicted_edges"], selected["true_edges"]) != edges:
            raise AssertionError(f"edge count mismatch for {net}")

        main_row = (
            f"{net} & {selected['seed']}  & {selected['validation_ce']:.6f} & "
            f"{selected['acc']:.4f} & {selected['precision']:.4f} & "
            f"{selected['f1']:.4f} & {selected['tpr']:.4f} & "
            f"{selected['fpr']:.4f} & {selected['predicted_edges']}/{selected['true_edges']}"
        ).replace("  &", " &")
        normalized_results = re.sub(r"\s+", "", results_tex)
        if re.sub(r"\s+", "", main_row) not in normalized_results:
            raise AssertionError(f"selected-model table is out of sync for {net}")

    for run in manifest["unknown_topology_all_runs"]:
        seed_cell = f"{run['seed']}$^*$" if run["selected"] else str(run["seed"])
        appendix_row = (
            f"{run['network']} & {seed_cell} & {run['validation_ce']:.6f} & "
            f"{run['acc']:.4f} & {run['precision']:.4f} & {run['f1']:.4f} & "
            f"{run['tpr']:.4f} & {run['predicted_edges']}/{run['true_edges']}"
        ).replace("  &", " &")
        normalized_appendix = re.sub(r"\s+", "", appendix_tex)
        if re.sub(r"\s+", "", appendix_row) not in normalized_appendix:
            raise AssertionError(
                f"all-seed table is out of sync for {run['network']} seed {run['seed']}"
            )

    known_values = manifest["known_topology"]
    for values in known_values.values():
        formatted = [
            f"{values['mean_abs_beta_error']:.5f}",
            f"{values['mean_per_k_mae']:.5f}",
            f"{values['mean_abs_gamma_error']:.5f}",
            f"{values['mean_delta_gamma']:.5f}",
            f"{values['mean_autoreg_mae_i']:.5f}",
        ]
        if not all(value in results_tex for value in formatted):
            raise AssertionError(f"known-topology summary is out of sync: {values['label']}")

    for name in (
        "research_pipeline.pdf",
        "architecture_compare.pdf",
        "joint_training_flow.pdf",
        "known_topology_results.pdf",
        "seed_selection_and_reconstruction.pdf",
        "calibration_and_autoregression.pdf",
    ):
        if not (PAPER / "figures" / name).is_file():
            raise AssertionError(f"missing generated figure: {name}")

    print(
        "PASS: source hashes, generated figures, phase4/phase6 tables, selected seeds and edge counts"
    )


if __name__ == "__main__":
    main()
