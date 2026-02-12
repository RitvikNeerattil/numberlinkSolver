# numberlinkSolver

Research workspace for training Numberlink heuristics with DeepXube.

This repository combines:
- a custom Numberlink domain implementation inside DeepXube (`DeepXube/deepxube/domains/numberlink.py`)
- the NumberLink environment package as a submodule (`NumberLink/`)
- a root training launcher (`train_numberlink.py`)

## What This Repo Does

- Integrates Numberlink state/action/goal modeling into DeepXube.
- Trains heuristic models for Numberlink using DeepXube's updater + search loop.
- Supports curriculum-style training and multiprocessing for larger runs.

## Repository Layout

- `train_numberlink.py`: Main training entrypoint for Numberlink experiments.
- `DeepXube/`: DeepXube codebase (submodule), including custom `numberlink` domain.
- `NumberLink/`: NumberLink environment package (submodule).
- `DOCUMENTATION/`: Local notes and API references.

## Setup

### 1. Clone with submodules

```bash
git clone --recurse-submodules <your-repo-url>
cd numberlinkSolver
```

If already cloned:

```bash
git submodule update --init --recursive
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate
```

### 3. Install local packages

```bash
pip install -e ./DeepXube
pip install -e ./NumberLink
```