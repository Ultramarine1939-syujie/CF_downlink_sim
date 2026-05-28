"""Shared project paths for Python entrypoints."""

from __future__ import annotations

from pathlib import Path

PYTHON_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PYTHON_DIR.parent

DATA_DIR = PROJECT_ROOT / "data"
TRAINING_DATA_DIR = DATA_DIR / "gnn_training"
TRAINING_DATA_GLOB = str(TRAINING_DATA_DIR / "*.mat")

MODEL_DIR = PROJECT_ROOT / "models"
DOCS_DIR = PROJECT_ROOT / "docs"
MAIN_DIR = PROJECT_ROOT / "main"
FIGURE_DIR = MAIN_DIR / "Imgs"
SIMULATION_DATA_DIR = MAIN_DIR / "SimulationData"


def project_path(*parts: str) -> Path:
    """Return an absolute path under the project root."""
    return PROJECT_ROOT.joinpath(*parts)
