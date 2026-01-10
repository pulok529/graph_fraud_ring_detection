# Graph-based Fraud Ring Detection - Starter Project

Creates a synthetic transaction graph and identifies suspicious communities using:
- graph feature extraction
- simple community detection / centrality heuristics

This is a practical starting point for graph fraud research.

## Quick start
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python build_graph.py
python detect_rings.py
```
