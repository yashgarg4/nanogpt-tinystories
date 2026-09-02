# Makes `src` a Python package so we can run modules with `python -m src.data`,
# `python -m src.train`, etc. Keeping it a package also lets modules import each
# other with clean relative imports (e.g. `from .config import GPTConfig`).
