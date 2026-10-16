#!/usr/bin/env python3
"""Compatibility entrypoint: execute the two authoritative figure notebooks."""
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

PAPER = Path(__file__).resolve().parents[1]
NOTEBOOKS = [
    PAPER / "notebooks" / "01_method_diagrams.ipynb",
    PAPER / "notebooks" / "02_experiment_figures.ipynb",
]


def main():
    for path in NOTEBOOKS:
        print(f"Executing: {path.name}", flush=True)
        nb = nbformat.read(path, as_version=4)
        nbformat.validate(nb)
        manager = KernelManager(kernel_name="python3")
        manager.kernel_spec.argv = [
            sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"
        ]
        client = NotebookClient(
            nb, km=manager, timeout=180, allow_errors=False,
            resources={"metadata": {"path": str(path.parent)}},
        )
        try:
            client.execute()
            nbformat.write(nb, path)
        finally:
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)
        print(f"Saved figures and previews: {path.name}", flush=True)
    print("PASS: both notebooks executed from fresh kernels")


if __name__ == "__main__":
    main()
