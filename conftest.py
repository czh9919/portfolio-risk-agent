# Empty conftest at the repo root so pytest prepends the project root to
# sys.path, making the top-level packages (risk, strategy, data, ...) importable
# when CI runs `pytest tests/` rather than `python -m pytest`.
